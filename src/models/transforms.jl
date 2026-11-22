# Model re-parameterizations:
#  - `nullspace_transform`: eliminate linear equality constraints by
#    substituting x = x₀ + Z y (Z = nullspace basis).
#  - `unconstrained_transform`: map box-constrained variables to
#    unconstrained ones with TransformVariables.jl bijections.

using TransformVariables: TransformVariables
const TV = TransformVariables

"""
    StanSimplex(n)

The Stan (MCMC) simplex transform, as a TransformVariables transformation:
the `n - 1` unconstrained reals `z` map to the open simplex via the
log-ratio/softmax bijection `y = softmax([z; 0])` (the last component
pinned to zero log-weight — Stan's current simplex transform), with inverse
`zᵢ = log(yᵢ / yₙ)`. TransformVariables' own `UnitSimplex` uses the
stick-breaking construction instead, which is a different
parameterization; [`unconstrained_transform`](@ref) uses this one. The
log-Jacobian correction is not tracked (the transform is intended for
optimization re-parameterization, not density computation).
"""
struct StanSimplex <: TV.VectorTransform
    n::Int
end

TV.dimension(t::StanSimplex) = t.n - 1

function TV.transform_with(
    flag::TV.LogJacFlag, t::StanSimplex, x::AbstractVector, index::Int
)
    n = t.n
    T = TV._ensure_float(eltype(x))
    y = similar(x, T, n)
    z = @view x[index:(index + n - 2)]
    m = maximum(z; init=zero(T))
    s = zero(T)
    @inbounds for i in 1:(n - 1)
        yi = exp(z[i] - m)
        y[i] = yi
        s += yi
    end
    yn = exp(-m)
    y[n] = yn
    s += yn
    @inbounds for i in 1:n
        y[i] /= s
    end
    ℓ = TV.logjac_zero(flag, T)
    return y, ℓ, index + n - 1
end

function TV.inverse_at!(x::AbstractVector, index::Int, t::StanSimplex, y::AbstractVector)
    n = t.n
    @inbounds for i in 1:(n - 1)
        x[index + i - 1] = log(y[i] / y[n])
    end
    return index + n - 1
end

function TV.inverse_eltype(t::StanSimplex, ::Type{T}) where {T<:AbstractVector}
    TV._ensure_float(eltype(T))
end

######################################################
# Zygote/ChainRules support for TransformVariables
######################################################

# TransformVariables' `transform`/`inverse` carry ChainRules `rrule`s
# computed with forward-mode AD through the transformation, so Zygote
# gradients flow through models whose variables were re-parameterized by
# `unconstrained_transform`.
function ChainRulesCore.rrule(
    ::typeof(TV.transform), t::TV.AbstractTransform, z::AbstractVector
)
    y = TV.transform(t, z)
    proj = ChainRulesCore.ProjectTo(z)
    function transform_pullback(Δy)
        Δy = ChainRulesCore.unthunk(Δy)
        J = ForwardDiff.jacobian(zz -> _flat_out(TV.transform(t, zz)), z)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
        proj(J' * _flat_out(Δy))
    end
    return y, transform_pullback
end

function ChainRulesCore.rrule(::typeof(TV.inverse), t::TV.AbstractTransform, x)
    z = TV.inverse(t, x)
    function inverse_pullback(Δz)
        Δz = ChainRulesCore.unthunk(Δz)
        J = ForwardDiff.jacobian(xx -> _flat_out(TV.inverse(t, xx)), _flat_out(x))
        x̄ = J' * _flat_out(Δz)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(),
        x isa Tuple ? Tuple(x̄) : x̄
    end
    return z, inverse_pullback
end

# A (possibly nested, tuple-valued) transformation output flattened to a
# plain vector.
_flat_out(y::Tuple) = reduce(vcat, (_flat_out(x) for x in y); init=Float64[])
_flat_out(y::AbstractVector) = collect(y)
_flat_out(y::Number) = [y]
_flat_out(y::ChainRulesCore.AbstractThunk) = _flat_out(ChainRulesCore.unthunk(y))
_flat_out(y::ChainRulesCore.Tangent) = _flat_out(ChainRulesCore.backing(y))

# The inverse-space input of a TransformTuple: mirrors the (nested) output
# structure — vector-valued components (the simplex) take vector entries.
function _inverse_input(x0::AbstractVector, components)
    out = Any[]
    i = 1
    for t in components
        d = t isa StanSimplex ? t.n : 1
        if t isa StanSimplex
            push!(out, x0[i:(i + d - 1)])
        else
            push!(out, x0[i])
        end
        i += d
    end
    return Tuple(out)
end

######################################################
# Nullspace transform: eliminate linear equalities
######################################################

# Collect the rows (A, rhs) of every linear equality constraint of a model:
# legacy `EqConstraint`s with a LinearFunction and `FunctionInSet` constraints
# whose function is a LinearFunction and whose set is EqualTo (plain, not
# inside a product).
function _linear_equality_rows(m::VecModel)
    rows = Matrix{Float64}[]
    rhs = Float64[]
    for c in m.eq_constraints.fs
        raw = c.f isa FlattenedFunction ? getrawfunction(c.f) : c.f
        raw = raw isa FunctionWrapper ? getfunction(raw) : raw
        if raw isa LinearFunction
            # value = A x + b - c.rhs = 0  ⟺  A x = c.rhs - b
            push!(rows, Matrix{Float64}(raw.A))
            r = c.rhs isa AbstractVector ? collect(c.rhs) : fill(c.rhs, size(raw.A, 1))
            append!(rhs, r .- linear_offset(raw))
        end
    end
    for c in m.set_constraints.fs
        if c.set isa EqualTo && getfunction(c) isa LinearFunction
            lf = getfunction(c)
            push!(rows, Matrix{Float64}(lf.A))
            v = if c.set.value isa AbstractVector
                collect(c.set.value)
            else
                fill(c.set.value, size(lf.A, 1))
            end
            append!(rhs, v .- linear_offset(lf))
        end
    end
    isempty(rows) && return nothing
    n = length(getmin(m))
    A = reduce(vcat, rows)
    length(rhs) == size(A, 1) || throw(
        DimensionMismatch(
            "Linear equality rows are inconsistent: $(size(A, 1)) rows, " *
            "$(length(rhs)) right-hand sides.",
        ),
    )
    size(A, 2) == n || throw(
        DimensionMismatch(
            "Linear equalities act on $(size(A, 2)) variables but the model has $n."
        ),
    )
    return A, rhs
end

"""
    nullspace_transform(m::VecModel; tol = 1e-10)

Re-parameterizes the model to eliminate its linear equality constraints:
with `A x = b` the stacked linear equalities (first-class
[`LinearFunction`](@ref)s in [`EqualTo`](@ref) sets or legacy equality
constraints), the substitution `x = x₀ + Z y` (SVD nullspace basis `Z`,
particular solution `x₀`) yields an equivalent model in the reduced
variables `y` with no linear equalities.

Everything is composed with the affine map: the objective, the remaining
constraints and the box bounds — the box becomes first-class linear
inequality constraints on `y`, and variable-in-set blocks become
function-in-set constraints on `y` (their box parts inside the box
inequalities, their cone parts composed as set constraints). Returns
`(reduced_model, z0, mapper)` where `z₀ = Zᵀ(x₀ - x_part)` is the initial
reduced point and `mapper(z) = x_part + Z*z` maps a reduced solution back to
a solution of the original model (always returned; the identity when there
is nothing to eliminate). Integer variables are rejected (integrality does
not survive the re-parameterization).
"""
function nullspace_transform(m::VecModel; tol=1e-10)
    rows = _linear_equality_rows(m)
    rows === nothing && return m, getinit(m), identity
    A, rhs = rows
    any(m.integer) && throw(
        ArgumentError(
            "nullspace_transform cannot handle integer variables: the " *
            "re-parameterization does not preserve integrality.",
        ),
    )
    n = length(getmin(m))
    F = svd(A; full=true)
    r = findlast(s -> s > tol * max(F.S[1], 1.0), F.S)
    r === nothing && (r = 0)
    Z = Matrix{Float64}(F.V[:, (r + 1):end])
    x_part = if r == 0
        zeros(n)
    else
        F.V[:, 1:r] * ((F.U' * rhs)[1:r] ./ F.S[1:r])
    end
    ψ(y) = x_part .+ Z * y

    z0 = Z' * (getinit(m) - x_part)

    # The box bounds of x become linear inequalities on y:
    #   Z y ≤ u - x₀  and  Z y ≥ l - x₀.
    box_ineqs = IneqConstraint[]
    fin_u = findall(isfinite, getmax(m))
    if !isempty(fin_u)
        push!(
            box_ineqs,
            IneqConstraint(
                LinearFunction(Z[fin_u, :], x_part[fin_u] .- getmax(m)[fin_u]), 0.0
            ),
        )
    end
    fin_l = findall(isfinite, getmin(m))
    if !isempty(fin_l)
        push!(
            box_ineqs,
            IneqConstraint(
                LinearFunction(-Z[fin_l, :], getmin(m)[fin_l] .- x_part[fin_l]), 0.0
            ),
        )
    end
    append!(
        box_ineqs,
        IneqConstraint[
            IneqConstraint(y -> c(ψ(y)), c.rhs, c.dim, c.flags) for
            c in m.ineq_constraints.fs
        ],
    )

    rm = VecModel(
        if m.objective === nothing
            nothing
        else
            Objective(y -> m.objective(ψ(y)), m.objective.multiple, m.objective.flags)
        end,
        VectorOfFunctions(
            EqConstraint[
                EqConstraint(y -> c(ψ(y)), c.rhs, c.dim, c.flags) for
                c in m.eq_constraints.fs
            ],
        ),
        VectorOfFunctions(box_ineqs),
        VectorOfFunctions(
            SDConstraint[SDConstraint(y -> c(ψ(y)), c.dim) for c in m.sd_constraints.fs]
        ),
        VectorOfFunctions(
            FunctionInSet[
                FunctionInSet(y -> c(ψ(y)), c.set, c.flags) for c in m.set_constraints.fs
            ],
        ),
        VariableInSet[],
        fill(-Inf, size(Z, 2)),
        fill(Inf, size(Z, 2)),
        float.(z0),
        falses(size(Z, 2)),
    )
    return rm, float.(z0), ψ
end

######################################################
# Domain transform: constrained → unconstrained variables
######################################################

"""
    unconstrained_transform(m::VecModel; labels = nothing)

Re-parameterizes the model so that box-constrained variables become
unconstrained ones: each transformed variable with finite bounds `[l, u]` is
replaced by `zᵢ ∈ ℝ` with `xᵢ = l + (u - l) ⋅ logistic(zᵢ)` (the
TransformVariables.jl `asℝ(l, u)` bijection; unbounded variables pass
through identically). All model functions — objective, constraints, and the
box itself — are composed with the bijection, so the returned model has no
bounds on the transformed variables. Set-variable cone parts (PSD-ness, the
simplex sum) remain set constraints on the re-parameterized variables, and
the `transform`/`inverse` ChainRules rrules keep the model differentiable
under Zygote.

`labels` selects which variables to transform: with `labels = nothing` every
box-bounded variable is transformed; otherwise only the entries of the
[`VariableInSet`](@ref) blocks whose `flags` contain one of the labels are
(the other variables keep their bounds unchanged).

Returns `(transformed_model, z0, mapper)` where
`z₀ = inverse(t, x₀)` is the unconstrained initial point and
`mapper(z) = transform(t, z)` maps a solution of the transformed model back
to a solution of the original model (always returned; the identity when
nothing is transformed). Integer variables are rejected.
"""
function unconstrained_transform(m::VecModel; labels=nothing)
    any(m.integer) && throw(
        ArgumentError(
            "unconstrained_transform cannot handle integer variables: the " *
            "re-parameterization does not preserve integrality.",
        ),
    )
    n = length(getmin(m))
    l = getmin(m)
    u = getmax(m)
    sel_blocks = _selected_blocks(m, labels)
    if isempty(sel_blocks)
        return m, getinit(m), identity
    end
    simplex_starts = Dict(
        first(v.range) => v.set.dim for v in sel_blocks if v.set isa Simplex
    )
    rng_mask = _block_ranges(sel_blocks, n)
    transformed = falses(n)
    components = TV.AbstractTransform[]
    l_z = Float64[]
    u_z = Float64[]
    i = 1
    while i <= n
        if haskey(simplex_starts, i)
            # A simplex block becomes n-1 unconstrained reals through the
            # Stan log-ratio/softmax transform (the box and the sum-to-one
            # are satisfied by construction).
            d = simplex_starts[i]
            push!(components, StanSimplex(d))
            transformed[i:(i + d - 1)] .= true
            append!(l_z, fill(-Inf, d - 1))
            append!(u_z, fill(Inf, d - 1))
            i += d
        elseif rng_mask[i] && isfinite(l[i]) && isfinite(u[i])
            push!(components, TV.as(Real, l[i], u[i]))
            transformed[i] = true
            push!(l_z, -Inf)
            push!(u_z, Inf)
            i += 1
        else
            # Identity: the variable keeps its original bounds.
            push!(components, TV.asℝ)
            push!(l_z, l[i])
            push!(u_z, u[i])
            i += 1
        end
    end
    any(transformed) || return m, getinit(m), identity
    t = TV.as(Tuple(components))
    n_z = TV.dimension(t)
    x0 = getinit(m)
    z0 = _flat_out(TV.inverse(t, _inverse_input(x0, components)))
    ψ(y) = _flat_out(TV.transform(t, y))

    um = VecModel(
        if m.objective === nothing
            nothing
        else
            Objective(y -> m.objective(ψ(y)), m.objective.multiple, m.objective.flags)
        end,
        VectorOfFunctions(
            EqConstraint[
                EqConstraint(y -> c(ψ(y)), c.rhs, c.dim, c.flags) for
                c in m.eq_constraints.fs
            ],
        ),
        VectorOfFunctions(
            IneqConstraint[
                IneqConstraint(y -> c(ψ(y)), c.rhs, c.dim, c.flags) for
                c in m.ineq_constraints.fs
            ],
        ),
        VectorOfFunctions(
            SDConstraint[SDConstraint(y -> c(ψ(y)), c.dim) for c in m.sd_constraints.fs]
        ),
        VectorOfFunctions(
            FunctionInSet[
                FunctionInSet(y -> c(ψ(y)), c.set, c.flags) for c in m.set_constraints.fs
            ],
        ),
        VariableInSet[],
        float.(l_z),
        float.(u_z),
        float.(z0),
        falses(n_z),
    )
    return um, float.(z0), ψ
end

# The variable blocks selected by `labels`: all of them when `labels` is
# `nothing`, otherwise those whose `flags` contain one of the labels.
function _selected_blocks(m::VecModel, labels)
    labels === nothing && return m.set_variables
    lbls = Set{Symbol}(Symbol.(labels))
    return [v for v in m.set_variables if !isempty(intersect(v.flags, lbls))]
end

# The union of the flat entry ranges of a collection of variable blocks, as
# a Bool mask of length `n`.
function _block_ranges(blocks, n::Int)
    r = falses(n)
    for v in blocks
        r[v.range] .= true
    end
    return r
end
