
"""
    abstract type AbstractSet end

Abstract supertype of the constraint sets used in function-in-set constraints
([`FunctionInSet`](@ref)) and variable-in-set blocks ([`VariableInSet`](@ref)).
Concrete sets — "a function (or variable) constrained to lie in a set" —
are the [`VectorSet`](@ref) family, the semidefinite cone `PSDCone` and
the product set [`SetProduct`](@ref).
"""
abstract type AbstractSet end

# Sets compare structurally (field-wise), not by identity.
function Base.:(==)(a::T, b::T) where {T<:AbstractSet}
    return all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end

"""
    abstract type VectorSet <: AbstractSet end

A set on vectors of `ℝ^dim`: the domain of the order sets ([`EqualTo`](@ref),
[`LessThan`](@ref), [`GreaterThan`](@ref), [`Interval`](@ref)), the sign
cones ([`Nonnegatives`](@ref), [`Nonpositives`](@ref), [`Zeros`](@ref)),
[`Simplex`](@ref) and the general default [`Reals`](@ref).
"""
abstract type VectorSet <: AbstractSet end

"""
    Reals(dim)

The whole space `ℝ^dim`: the general default vector variable set, giving
`dim` free (unbounded) variables. The plain `addvar!` box API falls back on
the [`Interval`](@ref) vector set.
"""
struct Reals <: VectorSet
    dim::Int
end

"""
    EqualTo(value)

The set of functions equal to `value`. `value` can be a scalar (one output)
or a vector (several outputs, all pinned to the corresponding entries of
`value`).
"""
struct EqualTo{T} <: VectorSet
    value::T
    dim::Int
end
EqualTo(v::Real) = EqualTo(v, 1)
EqualTo(v::AbstractVector) = EqualTo(v, length(v))

"""
    LessThan(upper)

The set of functions less than or equal to `upper` component-wise.
"""
struct LessThan{T} <: VectorSet
    upper::T
    dim::Int
end
LessThan(u::Real) = LessThan(u, 1)
LessThan(u::AbstractVector) = LessThan(u, length(u))

"""
    GreaterThan(lower)

The set of functions greater than or equal to `lower` component-wise.
"""
struct GreaterThan{T} <: VectorSet
    lower::T
    dim::Int
end
GreaterThan(l::Real) = GreaterThan(l, 1)
GreaterThan(l::AbstractVector) = GreaterThan(l, length(l))

"""
    Interval(lower, upper)

The set of functions between `lower` and `upper` component-wise.
"""
struct Interval{T} <: VectorSet
    lower::T
    upper::T
    dim::Int
end
Interval(l::Real, u::Real) = Interval(promote(l, u)..., 1)
function Interval(l::AbstractVector, u::AbstractVector)
    length(l) == length(u) || throw(
        DimensionMismatch(
            "Interval bounds must have equal lengths, got $(length(l)) and $(length(u)).",
        ),
    )
    l2, u2 = promote(l, u)
    return Interval(l2, u2, length(l2))
end

"""
    PSDCone(n)

The cone of positive semidefinite `n × n` symmetric matrices. The function
evaluated against it must return a matrix; a variable block registered with
[`add_set_variable!`](@ref) is an `n × n` semidefinite matrix variable. The
chordal-decomposition and semidefinite algorithms live in
NonconvexSemidefinite.jl; the cone type itself is defined here so that all
sets share one home.
"""
struct PSDCone <: AbstractSet
    dim::Int
end

"""
    Nonnegatives(dim)

The cone of vectors with nonnegative entries.
"""
struct Nonnegatives <: VectorSet
    dim::Int
end

"""
    Nonpositives(dim)

The cone of vectors with nonpositive entries.
"""
struct Nonpositives <: VectorSet
    dim::Int
end

"""
    Zeros(dim)

The set consisting of the zero vector of length `dim`.
"""
struct Zeros <: VectorSet
    dim::Int
end

"""
    Simplex(dim)

The probability simplex `{x ∈ ℝ^dim : x ≥ 0, Σᵢ xᵢ = 1}`.
"""
struct Simplex <: VectorSet
    dim::Int
end

"""
    Zero()

A special value for the probability input of a [`ChanceSet`](@ref): with
`epsilon = Zero()` the constraint is deterministic — the function is
evaluated exactly once with `ξ = nothing` and must lie in the inner set
(probability one). The scenario batch is ignored.
"""
struct Zero end

"""
    ChanceSet(set, scenarios; epsilon = 0.05, kappa = 10.0)

A probabilistic constraint set: a function `f(x, ξ)` lies in the inner
`set` **with a certain probability** — `P_ξ[f(x, ξ) ∈ set] ≥ 1 - epsilon`,
where the exceedance probability is estimated by a smooth (tanh) sample
average of the exceedance indicator with sharpness `kappa`, which is
differentiable and exactly zero on feasible scenarios.

The scenario batch `scenarios` (a vector of `ξ` values) is supplied by the
caller: NonconvexCore defines the constraint and evaluates it over the
given batch, it does not draw random scenarios — the RNG (and the seed)
stays with the caller. With `epsilon = Zero()` the constraint is
deterministic and `scenarios` is ignored (`f` is evaluated once with
`ξ = nothing`).

When the inner set is a [`SetProduct`](@ref), `epsilon` can be a vector of
probabilities, one per output — each output lies in its own set with its
own probability (`P[fⱼ(x, ξ) ∈ Sⱼ] ≥ 1 - εⱼ`), all sharing the same
scenario batch.

`epsilon = Zero()` expresses a deterministic constraint: the single
evaluation is exact (no smoothing).

Use it in the function-in-set form — the [`FunctionInSet`](@ref) evaluation
iterates the scenario batch and returns the batch of inner-set values — or
through [`add_chance_constraint!`](@ref):

```julia
ξs = [rand(StableRNG(1), dist) for _ in 1:1000]
add_set_constraint!(m, FunctionInSet((x, ξ) -> x[1] + ξ,
                                     ChanceSet(LessThan(1.0), ξs; epsilon = 0.1)))
```
"""
struct ChanceSet{S<:AbstractSet,E} <: AbstractSet
    set::S
    epsilon::E
    kappa::Float64
    n_samples::Int
    scenarios::Vector
end
function ChanceSet(set::AbstractSet, scenarios; epsilon=0.05, kappa=10.0)
    if epsilon isa Zero
        return ChanceSet(set, Zero(), Float64(kappa), 1, Any[nothing])
    end
    scenarios isa AbstractVector || throw(
        ArgumentError(
            "Chance constraints take a vector of scenarios; draw them with the RNG of your choice.",
        ),
    )
    return ChanceSet(
        set,
        _epsilon_vector(set, epsilon),
        Float64(kappa),
        length(scenarios),
        collect(scenarios),
    )
end

# One probability per inner output: a vector of length k for a product of k
# sets, a 1-vector for a single set (a scalar epsilon broadcasts).
function _epsilon_vector(set::AbstractSet, epsilon)
    if epsilon isa AbstractVector
        length(epsilon) == length(components(set)) || throw(
            DimensionMismatch(
                "Got $(length(epsilon)) probabilities for $(length(components(set))) " *
                "output sets; provide one probability per output.",
            ),
        )
        return Float64.(epsilon)
    end
    return fill(Float64(epsilon), length(components(set)))
end

"""
    InequalitySet(upper)  (alias of [`LessThan`](@ref))

The default set of the classical inequality-constraint syntax: the set of
functions `f` with `f(x) ≤ upper`. `add_ineq_constraint!(m, f, s)` is
`f ∈ InequalitySet(s)` under the hood; use [`add_set_constraint!`](@ref) for
the general function-in-set form.
"""
const InequalitySet = LessThan

"""
    EqualitySet(value)  (alias of [`EqualTo`](@ref))

The default set of the classical equality-constraint syntax: the set of
functions `f` with `f(x) = value`. `add_eq_constraint!(m, f, s)` is
`f ∈ EqualitySet(s)` under the hood.
"""
const EqualitySet = EqualTo

"""
    SetProduct(sets...)
    SetProduct(sets::AbstractVector{<:AbstractSet})

The product of sets. A single function `x -> (v₁, v₂, …, vₖ)` evaluated
against `SetProduct(s₁, s₂, …, sₖ)` lies in the set when `vᵢ` lies in `sᵢ`
for every component. This is the construct that lets one expensive
computation (e.g. a finite-element displacement solve) feed several
constraints of different kinds — scalar, vector and matrix-valued — with a
single function evaluation.

Components can be composed freely: `SetProduct(Simplex(4), Simplex(4))` is
a pair of simplexes, and products may be nested. Semidefinite components
(`PSDCone`) compose the
same way.
"""
struct SetProduct{S<:Tuple{Vararg{AbstractSet}}} <: AbstractSet
    sets::S
end
SetProduct(sets::AbstractSet...) = SetProduct(sets)
SetProduct(sets::AbstractVector{<:AbstractSet}) = SetProduct(sets...)

"""
    getdim(s::AbstractSet)

The dimension of a set: the number of scalar entries the function value
contributes. Vector sets contribute their length, a semidefinite cone (`PSDCone(n)`)
contributes `n²` (the entries of the matrix) and a product set contributes
the sum over its components.
"""
getdim(s::EqualTo) = s.dim
getdim(s::LessThan) = s.dim
getdim(s::GreaterThan) = s.dim
getdim(s::Interval) = s.dim
getdim(s::PSDCone) = s.dim * s.dim
getdim(s::Reals) = s.dim
getdim(s::Nonnegatives) = s.dim
getdim(s::Nonpositives) = s.dim
getdim(s::Zeros) = s.dim
getdim(s::Simplex) = s.dim
getdim(s::ChanceSet) = s.n_samples
getdim(s::SetProduct) = sum(getdim, s.sets; init=0)

"""
    components(s::AbstractSet)

The component sets of `s`: the set itself for an atomic set, the tuple of
component sets for a [`SetProduct`](@ref).
"""
components(s::AbstractSet) = (s,)
components(s::SetProduct) = s.sets

# Output-shape validation of a function value against a set.
function check_set_output(value, set::AbstractSet)
    if set isa SetProduct
        (value isa Tuple || value isa AbstractVector) &&
        length(value) == length(set.sets) || throw(
            ArgumentError(
                "A function evaluated against a SetProduct of $(length(set.sets)) sets " *
                "must return a tuple or vector of the same length, one value per " *
                "set. Got a value of type $(typeof(value)).",
            ),
        )
        for (v, s) in zip(value, set.sets)
            check_set_output(v, s)
        end
        return nothing
    end
    # Vector sets: a vector of the right length.
    vector_sets = Union{Nonnegatives,Nonpositives,Zeros,Simplex}
    if set isa vector_sets
        value isa AbstractVector && length(value) == set.dim || throw(
            ArgumentError(
                "A function evaluated against $(typeof(set))($(set.dim)) must return a " *
                "vector of length $(set.dim), got $(typeof(value)).",
            ),
        )
        return nothing
    end
    if set isa ChanceSet
        value isa AbstractVector && length(value) == set.n_samples || throw(
            ArgumentError(
                "A function evaluated against a ChanceSet with $(set.n_samples) scenarios " *
                "must return a batch (vector) of the same length, got $(typeof(value)).",
            ),
        )
        for b in value
            check_set_output(b, set.set)
        end
        return nothing
    end
    # Order sets: scalar bounds take a scalar value; vector bounds take a
    # vector of the same length (a length-1 vector bound is still a vector
    # set). Other set kinds (e.g. the semidefinite cone PSDCone provided by
    # NonconvexSemidefinite.jl, or `Reals`, which is vacuous) fall through
    # to the extension hook below.
    if set isa Union{EqualTo,LessThan,GreaterThan,Interval}
        boundvec = if set isa EqualTo
            set.value isa AbstractVector
        elseif set isa LessThan
            set.upper isa AbstractVector
        elseif set isa GreaterThan
            set.lower isa AbstractVector
        else # Interval
            set.lower isa AbstractVector
        end
        if boundvec
            value isa AbstractVector && length(value) == set.dim || throw(
                ArgumentError(
                    "A function evaluated against a $(set.dim)-dimensional vector set must " *
                    "return a vector of length $(set.dim), got $(typeof(value)).",
                ),
            )
        else
            value isa Real || throw(
                ArgumentError(
                    "A function evaluated against a scalar set must return a real " *
                    "number, got $(typeof(value)).",
                ),
            )
        end
    end
    # Extension hook for set kinds defined outside NonconvexCore.
    _check_set_output(value, set)
    return nothing
end

_check_set_output(value, set::AbstractSet) = nothing

function _check_set_output(value, s::PSDCone)
    value isa AbstractMatrix && size(value) == (s.dim, s.dim) || throw(
        ArgumentError(
            "A function evaluated against PSDCone($(s.dim)) must return a " *
            "$(s.dim)×$(s.dim) matrix, got $(typeof(value)).",
        ),
    )
    return nothing
end

"""
    FunctionInSet(f, set; flags = Set{Symbol}())

A constraint of the form "the function `f` lies in the set `set`", i.e.
`f(x) ∈ set`. `f` is evaluated once per
call and its output is checked against `set`:

- `EqualTo(value)`/`LessThan(upper)`/`GreaterThan(lower)`/`Interval(lo, up)`
  accept scalar or vector-valued `f`;
- `PSDCone(n)` accepts a
  matrix-valued `f` (the matrix must be positive semidefinite);
- `SetProduct(s₁, …, sₖ)` accepts a `k`-tuple output, routing each component
  to its own set — a single evaluation can therefore feed e.g. an inequality
  and a semidefinite constraint at once, reusing the computation inside `f`.

`flags` labels the constraint: meta-algorithms select constraints by label,
e.g. `chordal_transform(m, patterns; labels = [:chordal])` (from
NonconvexSemidefinite.jl) and
`AugLagOptions(relaxed = RelaxConstraintSelection(labels = [:relax]))`.

Add one to a model with [`add_set_constraint!`](@ref).
"""
struct FunctionInSet{F,S<:AbstractSet} <: AbstractConstraint
    f::F
    set::S
    flags::Set{Symbol}
end
function FunctionInSet(f, set::AbstractSet; flags=Set{Symbol}())
    return FunctionInSet(f, set, _flags_set(flags))
end

function (c::FunctionInSet)(x)
    # A ChanceSet evaluates its two-argument function over the fixed Monte
    # Carlo scenarios: "the function lies in the inner set with a certain
    # probability".
    out = if c.set isa ChanceSet
        [c.f(x, ξ) for ξ in c.set.scenarios]
    else
        c.f(x)
    end
    check_set_output(out, c.set)
    return out
end

"""
    chance(c::AbstractConstraint, scenarios; epsilon = 0.05, kappa = 10.0)

The probabilistic version of any existing constraint `c`: the
[`FunctionInSet`](@ref) constraint
`P_ξ[c.f(x, ξ) ∈ setof(c)] ≥ 1 - epsilon` over the user-supplied scenario
batch `scenarios` (see [`ChanceSet`](@ref)). The classical constraints —
`IneqConstraint` (`f ≤ s`), `EqConstraint` (`f = s`), `SDConstraint`
(`f ⪰ 0`) — as well as any `FunctionInSet` — can be turned probabilistic
this way.
"""
function chance(c::AbstractConstraint, scenarios; kwargs...)
    f = getfunction(c)
    set = c isa FunctionInSet ? c.set : setof(c)
    flags = _flags(c)
    return FunctionInSet(f, ChanceSet(set, scenarios; kwargs...), flags)
end

_flags(c) = hasfield(typeof(c), :flags) ? c.flags : Set{Symbol}()

getdim(c::FunctionInSet) = getdim(c.set)
getfunction(c::FunctionInSet) = c.f

# Evaluate the function of a set constraint once and split the output into
# its components (always as a tuple, one entry per component set). A vector
# output against a product of several sets is split into its entries.
function set_value(c::FunctionInSet, x)
    out = c(x)
    comps = components(c.set)
    if out isa Tuple
        return out
    elseif out isa AbstractVector && length(comps) > 1 && length(out) == length(comps)
        return Tuple(out)
    else
        return (out,)
    end
end

# A scalar/vector value as a plain 1-D vector.
_set_vec(v::Real) = [v]
_set_vec(v) = vec(v)

# Component-wise violation of a set: a vector of nonnegative scalars, all
# zero exactly when the function value lies in the set.
_component_violation(vi, s::EqualTo) = _set_vec(abs.(vi .- s.value))
_component_violation(vi, s::LessThan) = _set_vec(max.(vi .- s.upper, 0.0))
_component_violation(vi, s::GreaterThan) = _set_vec(max.(s.lower .- vi, 0.0))
function _component_violation(vi, s::Interval)
    return _set_vec(max.(s.lower .- vi, vi .- s.upper, 0.0))
end
function _component_violation(vi, s::PSDCone)
    λmin = minimum(eigvals(Symmetric(Matrix(vi))))
    return [max(0.0, -λmin)]
end
_component_violation(vi, s::Nonnegatives) = _set_vec(max.(-vi, 0.0))
_component_violation(vi, s::Nonpositives) = _set_vec(max.(vi, 0.0))
_component_violation(vi, s::Zeros) = _set_vec(abs.(vi))
function _component_violation(vi, s::Simplex)
    return [max.(-vi, 0.0)..., abs(sum(vi) - 1.0)]
end
function _component_violation(batch, s::ChanceSet)
    # Per-output exceedance probabilities: for each inner output j, p̂ⱼ is the
    # (smoothly indicated) sample average of the largest j-th component
    # violation — each output lies in its own set with its own probability
    # 1 - εⱼ. The deterministic (epsilon = Zero()) case uses exact
    # indicators.
    comps = components(s.set)
    exact = s.epsilon isa Zero
    return map(1:length(comps)) do j
        vs = map(batch) do b
            bj = if b isa Tuple
                b[j]
            elseif b isa AbstractVector && length(comps) > 1
                b[j]
            else
                b
            end
            vj = _component_violation(bj, comps[j])
            return isempty(vj) ? 0.0 : maximum(vj)
        end
        p̂ = if exact
            # Deterministic (epsilon = Zero()): the violation is the
            # inner-set violation itself, exactly (ε = 0).
            return maximum(vs)
        else
            sum(_soft_indicator(s.kappa * v) for v in vs) / length(vs)
        end
        return max(0.0, p̂ - s.epsilon[j])
    end
end
# A smooth surrogate of the exceedance indicator 1{v > 0}: tanh(κv/2) is
# smooth, differentiable, and exactly zero at v = 0 (so feasible scenarios
# contribute nothing to the exceedance estimate).
_soft_indicator(z::Real) = tanh(z / 2)
# The logistic function (defined locally to avoid an extra dependency).
_logistic(z::Real) = 1.0 / (1.0 + exp(-z))
function _component_violation(vi, s::SetProduct)
    (vi isa Tuple || vi isa AbstractVector) && length(vi) == length(s.sets) || throw(
        ArgumentError(
            "A function evaluated against a SetProduct of $(length(s.sets)) sets must " *
            "return a tuple or vector of the same length, got $(typeof(vi)).",
        ),
    )
    out = Float64[]
    for (v, c) in zip(vi, s.sets)
        append!(out, _component_violation(v, c))
    end
    return out
end

"""
    set_kinds(s::AbstractSet)

The constraint kinds a function-in-set constraint against `s` needs a solver
to support, as a vector of the symbols `:eq` (EqualTo/Zeros), `:ineq`
(LessThan/GreaterThan/Interval/Nonnegatives/Nonpositives), `:sd` (PSDCone),
`:simplex` (Simplex) and
`:product` (SetProduct, recursing into its components). `Reals` constraints
are vacuous (no kind). These map one-to-one to the granular `sets_*`
capabilities of [`ConstraintCapabilities`](@ref).
"""

set_kinds(s::Union{EqualTo,Zeros}) = [:eq]
set_kinds(s::Union{LessThan,GreaterThan,Interval,Nonnegatives,Nonpositives}) = [:ineq]
set_kinds(::PSDCone) = [:sd]
set_kinds(::Simplex) = [:simplex]
set_kinds(::ChanceSet) = [:chance]
set_kinds(::Reals) = Symbol[]
function set_kinds(s::SetProduct)
    out = Symbol[:product]
    for c in s.sets
        append!(out, set_kinds(c))
    end
    return unique(out)
end

"""
    has_psd_component(s::AbstractSet)

Whether the set `s` has a semidefinite component ([`PSDCone`](@ref)),
recursing into [`SetProduct`](@ref).
"""
has_psd_component(::PSDCone) = true
has_psd_component(s::SetProduct) = any(has_psd_component, s.sets)
has_psd_component(s::AbstractSet) = false

"""
    variable_set_kinds(s::AbstractSet)

The cone kinds of a variable block constrained to `s` that the box cannot
express — `:sd` for a `PSDCone` block, `:simplex` for a `Simplex` block —
recursing into [`SetProduct`](@ref)
components. Box-expressible sets (the order sets, sign cones, `Zeros`,
`Reals`) contribute nothing: they are covered by the `box` capability. These
map to the `variable_sets_*` capabilities of
[`ConstraintCapabilities`](@ref).
"""
variable_set_kinds(s::AbstractSet) = Symbol[]
variable_set_kinds(::PSDCone) = [:sd]
variable_set_kinds(::Simplex) = [:simplex]
variable_set_kinds(s::Union{EqualTo,LessThan,GreaterThan,Interval}) = Symbol[]
variable_set_kinds(s::Union{Nonnegatives,Nonpositives,Zeros,Reals,ChanceSet}) = Symbol[]
function variable_set_kinds(s::SetProduct)
    out = Symbol[]
    for c in s.sets
        append!(out, variable_set_kinds(c))
    end
    return unique(out)
end

"""
    set_violation(c::FunctionInSet, x)

The violation of a function-in-set constraint at `x`: a vector of nonnegative
scalars, all zero exactly when `c(x)` lies in the set. Order-set components
contribute one entry per output; each PSD component contributes
`max(0, -λmin(P))`.
"""
function set_violation(c::FunctionInSet, x)
    v, comps = set_value(c, x), components(c.set)
    vs = [_component_violation(vi, s) for (vi, s) in zip(v, comps)]
    return isempty(vs) ? Float64[] : reduce(vcat, vs)
end

######################################################
# Variables in sets
######################################################

"""
    VariableInSet(set, range; flags = Set{Symbol}())

A block of decision variables constrained to lie in the set `set`: the
entries of the (flattened) variable vector in `range` must form a value in
`set`. Created by [`add_set_variable!`](@ref) and [`addvar!`](@ref); a
`PSDCone(n)` block is an `n × n`
semidefinite matrix variable, a `Simplex(n)` block is a probability simplex
variable, and the order sets give bounded or fixed variable blocks. `flags`
labels the block for meta-algorithm selection.
"""
struct VariableInSet{S<:AbstractSet}
    set::S
    range::UnitRange{Int}
    flags::Set{Symbol}
end
function VariableInSet(set, range::UnitRange{Int}; flags=Set{Symbol}())
    return VariableInSet(set, range, _flags_set(flags))
end

# Normalize a label collection (Symbol, vector of Symbols, set, …) to a
# Set{Symbol}.
_flags_set(f::Set{Symbol}) = f
_flags_set(f::AbstractSet) = Set{Symbol}(f)
_flags_set(f::Symbol) = Set{Symbol}([f])
_flags_set(f) = Set{Symbol}(Symbol.(collect(f)))

"""
    variable_dim(set::AbstractSet)

The number of scalar decision variables a variable block constrained to
`set` introduces: `dim` for the vector sets, `n²` for a semidefinite cone
(`PSDCone(n)`), and the sum over the components of a [`SetProduct`](@ref).
"""
variable_dim(s::AbstractSet) = s.dim
variable_dim(s::PSDCone) = s.dim * s.dim
variable_dim(s::SetProduct) = sum(variable_dim, s.sets; init=0)

# Box bounds of the variable block of a set (the cone parts beyond the box
# are encoded as synthetic set constraints, see `variable_set_constraints`).
_vecval(v::AbstractVector) = collect(v)
_vecval(v::Real) = [v]
function variable_box(s::EqualTo)
    v = _vecval(s.value)
    return v, copy(v)
end
function variable_box(s::LessThan)
    return fill(-Inf, s.dim), _vecval(s.upper)
end
function variable_box(s::GreaterThan)
    return _vecval(s.lower), fill(Inf, s.dim)
end
function variable_box(s::Interval)
    return _vecval(s.lower), _vecval(s.upper)
end
variable_box(s::Nonnegatives) = (zeros(s.dim), fill(Inf, s.dim))
variable_box(s::Nonpositives) = (fill(-Inf, s.dim), zeros(s.dim))
variable_box(s::Zeros) = (zeros(s.dim), zeros(s.dim))
variable_box(s::Simplex) = (zeros(s.dim), fill(Inf, s.dim))
variable_box(s::PSDCone) = (fill(-Inf, s.dim^2), fill(Inf, s.dim^2))
function variable_box(s::SetProduct)
    lbs = Float64[]
    ubs = Float64[]
    for c in s.sets
        l, u = variable_box(c)
        append!(lbs, l)
        append!(ubs, u)
    end
    return lbs, ubs
end

# A default initial point strictly inside (the interior of) the set.
variable_init(s::EqualTo) = _vecval(s.value)

# An interior point of the box [l, u] (robust to infinite bounds).
function _box_init(l::AbstractVector, u::AbstractVector)
    out = clamp.(zeros(promote_type(eltype(l), eltype(u))), l, u)
    for i in eachindex(out)
        isfinite(u[i]) && out[i] == u[i] && l[i] < u[i] && (out[i] = u[i] - 1.0)
        isfinite(l[i]) && out[i] == l[i] && l[i] < u[i] && (out[i] = l[i] + 1.0)
    end
    return out
end
variable_init(s::LessThan) = _box_init(fill(-Inf, s.dim), _vecval(s.upper))
variable_init(s::GreaterThan) = _box_init(_vecval(s.lower), fill(Inf, s.dim))
variable_init(s::Interval) = _box_init(_vecval(s.lower), _vecval(s.upper))
variable_init(s::Reals) = zeros(s.dim)
variable_init(s::Nonnegatives) = ones(s.dim)
variable_init(s::Nonpositives) = -ones(s.dim)
variable_init(s::Zeros) = zeros(s.dim)
variable_init(s::Simplex) = fill(1.0 / s.dim, s.dim)
variable_init(s::PSDCone) = vec(Matrix{Float64}(I, s.dim, s.dim))
function variable_init(s::SetProduct)
    out = Float64[]
    for c in s.sets
        append!(out, variable_init(c))
    end
    return out
end

# Box bounds of a `Reals` variable block.
variable_box(s::Reals) = (fill(-Inf, s.dim), fill(Inf, s.dim))

"""
    setof(c::AbstractConstraint)

The set view of a classical constraint: the default set its syntax
corresponds to — `InequalitySet(rhs)` for an [`IneqConstraint`](@ref),
`EqualitySet(rhs)` for an [`EqConstraint`](@ref) and `PSDCone(dim)` for an
[`SDConstraint`](@ref). See also [`tosetconstraint`](@ref).
"""
setof(c::IneqConstraint) = InequalitySet(c.rhs)
setof(c::EqConstraint) = EqualitySet(c.rhs)
setof(c::SDConstraint) = PSDCone(c.dim)

"""
    tosetconstraint(c::AbstractConstraint)

The [`FunctionInSet`](@ref) form of a classical constraint: `c.f ∈ setof(c)`.
Provided as the migration path to the unified function-in-set model of the
upcoming NonconvexCore 3, where `add_ineq_constraint!(m, f, s)` stores
`FunctionInSet(f, InequalitySet(s))` directly.
"""
tosetconstraint(c::IneqConstraint) = FunctionInSet(c.f, setof(c), c.flags)
tosetconstraint(c::EqConstraint) = FunctionInSet(c.f, setof(c), c.flags)
tosetconstraint(c::SDConstraint) = FunctionInSet(c.f, setof(c))

"""
    variable_set_constraints(v::VariableInSet)

The synthetic [`FunctionInSet`](@ref) constraints that encode the cone parts
of a variable-in-set block that the box cannot express: the PSD constraint on the matrix block of a
`PSDCone` variable and the sum-to-one equality of a `Simplex` variable (its
nonnegativity lives in the box). Box-expressible sets produce no
constraints.
"""
function variable_set_constraints(v::VariableInSet)
    set = v.set
    out = FunctionInSet[]
    if set isa SetProduct
        off = first(v.range) - 1
        for s in set.sets
            n = variable_dim(s)
            r = (off + 1):(off + n)
            append!(out, variable_set_constraints(VariableInSet(s, r)))
            off += n
        end
    else
        c = _variable_cone_constraint(set, v.range)
        c === nothing || push!(out, c)
    end
    return out
end

# The cone constraint of a variable block that the box cannot express; set
# kinds defined outside NonconvexCore extend this hook with their own type.
_variable_cone_constraint(set::AbstractSet, r) = nothing
function _variable_cone_constraint(set::Simplex, r)
    return FunctionInSet(x -> sum(view(x, r)), EqualTo(1.0))
end
function _variable_cone_constraint(set::PSDCone, r)
    n = set.dim
    return FunctionInSet(x -> begin
        A = reshape(view(x, r), n, n)
        return (A .+ A') ./ 2
    end, PSDCone(n))
end
