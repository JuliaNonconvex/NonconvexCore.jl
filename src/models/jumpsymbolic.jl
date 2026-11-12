# Functions whose derivatives come from MathOptInterface's Nonlinear
# evaluator (the stack JuMP uses for its new nonlinear interface), used for
# JuMP models converted into NonconvexCore models.

_unthunk(x) = x
_unthunk(x::ChainRulesCore.AbstractThunk) = ChainRulesCore.unthunk(x)

"""
    JuMPSymbolicFunction(expr::MOI.AbstractScalarFunction, vars::Vector{MOI.VariableIndex})

Returns a callable that evaluates `expr` at a flat vector `x` (ordered like
`vars`) with value, gradient and Hessian computed by MathOptInterface's
`Nonlinear` sparse reverse-mode evaluator. Reverse-mode differentiation of
the callable uses a ChainRules `rrule`; the gradient and Hessian are also
available via [`jump_symbolic_gradient`](@ref) and
[`jump_symbolic_hessian`](@ref).
"""
struct JuMPSymbolicFunction{E,X,V} <: Function
    evaluator::E
    expr::X
    vars::V
end

function JuMPSymbolicFunction(
    expr::MOI.AbstractScalarFunction, vars::Vector{MOI.VariableIndex}
)
    model = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(model, expr)
    evaluator = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    MOI.initialize(evaluator, [:Grad, :Hess])
    return JuMPSymbolicFunction(evaluator, expr, vars)
end

(f::JuMPSymbolicFunction)(x) = MOI.eval_objective(f.evaluator, x)

"""
    jump_symbolic_hessian_sparsity(f::JuMPSymbolicFunction, n)

Returns the sparse Hessian structure `(row, column)` of the wrapped
expression with respect to a vector of length `n`.
"""
function jump_symbolic_hessian_sparsity(f::JuMPSymbolicFunction, n)
    positions = Dict(v.value => k for (k, v) in enumerate(f.vars))
    return [
        (positions[i], positions[j]) for
        (i, j) in MOI.hessian_objective_structure(f.evaluator)
    ]
end

"""
    jump_symbolic_gradient(f::JuMPSymbolicFunction, x)

Returns the gradient of the wrapped function at `x`, computed by the
MathOptInterface Nonlinear evaluator.
"""
function jump_symbolic_gradient(f::JuMPSymbolicFunction, x)
    g = zeros(length(x))
    MOI.eval_objective_gradient(f.evaluator, g, x)
    return g
end

"""
    jump_symbolic_hessian(f::JuMPSymbolicFunction, x)

Returns the Hessian of the wrapped function at `x`, computed by the
MathOptInterface Nonlinear evaluator.
"""
function jump_symbolic_hessian(f::JuMPSymbolicFunction, x)
    structure = jump_symbolic_hessian_sparsity(f, length(x))
    vals = zeros(length(structure))
    MOI.eval_hessian_objective(f.evaluator, vals, x)
    H = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(structure)
        H[i, j] = vals[k]
        H[j, i] = vals[k]
    end
    return H
end

function ChainRulesCore.rrule(f::JuMPSymbolicFunction, x)
    y = f(x)
    return y, Δ -> (NoTangent(), jump_symbolic_gradient(f, x) * _unthunk(Δ))
end

"""
    JuMPConvertedRow

One converted constraint row of a JuMP model, in the original JuMP constraint
order. Inequality rows hold `sign * f(x) + shift` (feasible when <= 0);
equality rows hold `f(x) + shift`. Exactly one of `ci` (the row lives in the
MathOptInterface Nonlinear model) and `expr` (the row is affine or quadratic
and is evaluated in closed form) is non-`nothing`.
"""
struct JuMPConvertedRow
    is_ineq::Bool
    sign::Float64
    shift::Float64
    linear::Bool
    ci::Union{Nothing,MOI.Nonlinear.ConstraintIndex}
    expr::Union{Nothing,MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction}
end

"""
    JuMPEvaluator

The derivative oracle of a JuMP model converted into a NonconvexCore model.
The objective and every constraint are expressions of one
`MOI.Nonlinear.Model` evaluated through MathOptInterface's sparse
reverse-mode evaluator, so shared subexpressions are evaluated once per
pass and the whole Jacobian and Lagrangian Hessian come from single
evaluator calls. Affine and quadratic rows (and an affine or quadratic
objective) are evaluated in closed form instead of through the evaluator,
matching how JuMP itself splits such functions out of its nonlinear block;
their values, Jacobians and Hessians are exact and evaluate in O(nnz).

Converted rows are exposed inequality-first (in the JuMP constraint order),
then equalities (also in the JuMP constraint order). The constraints are
keyed by their `ConstraintIndex`, because the evaluator row order is its
own; the evaluator's row order is recovered from `ordered_constraints`.
"""
mutable struct JuMPEvaluator{E,V}
    ev::E
    vars::V
    objective_sign::Float64
    nineq::Int
    neq::Int
    jac_rows::Vector{Int}      # converted rows, aligned with the jacobian values
    jac_cols::Vector{Int}
    jac_evrows::Vector{Int}    # evaluator rows of the nonlinear jacobian entries
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    ev_sign::Vector{Float64}   # per evaluator row
    ev_shift::Vector{Float64}  # per evaluator row
    ineq_linear::Vector{Bool}  # per converted row
    eq_linear::Vector{Bool}    # per converted row
    myrow_of_evrow::Vector{Int}
    g_buffer::Vector{Float64}   # nonlinear constraint values in evaluator row order
    jac_buffer::Vector{Float64} # nonlinear jacobian values in evaluator structure order
    hess_buffer::Vector{Float64} # nonlinear hessian values in evaluator structure order
    μ_buffer::Vector{Float64}   # Lagrangian multipliers in evaluator row order
    # Closed-form channel: affine and quadratic rows, evaluated analytically.
    cl_rows::Vector{Int}        # converted row of each closed-form row
    cl_sign::Vector{Float64}
    cl_shift::Vector{Float64}
    cl_q_id::Vector{Int}        # closed-form row id per quadratic term
    cl_q_a::Vector{Int}         # variable columns of the term
    cl_q_b::Vector{Int}
    cl_q_v::Vector{Float64}
    cl_c_id::Vector{Int}        # closed-form row id per affine term
    cl_c_col::Vector{Int}
    cl_c_v::Vector{Float64}
    cl_d::Vector{Float64}       # constant per closed-form row
    cl_g_buffer::Vector{Float64}
    # Merged Jacobian: positions and evaluation recipes of the closed rows.
    nl_jac_pos::Vector{Int}     # merged position of each nonlinear jacobian entry
    cl_jac_pos::Vector{Int}     # merged position of each closed jacobian entry
    cl_jac_terms::Vector{Vector{Tuple{Int,Float64}}} # (variable column, coef) per entry
    cl_jac_const::Vector{Float64}                    # constant per entry
    cl_jac_sign::Vector{Float64}                     # row sign per entry
    # Merged Hessian: positions and coefficients of every channel.
    nl_hess_pos::Vector{Int}    # merged position of each nonlinear hessian entry
    cl_hess_pos::Vector{Int}    # merged position per closed-form quadratic term
    cl_hess_myrow::Vector{Int}  # converted row per closed-form quadratic term
    cl_hess_coef::Vector{Float64}
    obj_hess_pos::Vector{Int}   # merged position per objective quadratic term
    obj_hess_coef::Vector{Float64}
    # Objective closed form (empty when the objective is nonlinear).
    obj_closed::Bool
    obj_q_a::Vector{Int}
    obj_q_b::Vector{Int}
    obj_q_v::Vector{Float64}
    obj_c::Vector{Float64}      # dense linear coefficients
    obj_d::Float64
end

function JuMPEvaluator(
    model::MOI.Nonlinear.Model,
    vars::Vector{MOI.VariableIndex},
    objective_sign::Float64,
    rows::Vector{JuMPConvertedRow},
    obj_closed::Union{Nothing,MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction},
)
    ev = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    features = [:Grad, :Hess]
    any(r -> r.ci !== nothing, rows) && push!(features, :Jac)
    MOI.initialize(ev, features)
    positions = Dict(v.value => k for (k, v) in enumerate(vars))
    nvars = length(vars)
    # Converted rows: inequality-first, equality-second, each in the original
    # JuMP constraint order carried by `rows`.
    ci_to_myrow = Dict{MOI.Nonlinear.ConstraintIndex,Int}()
    ci_sign = Dict{MOI.Nonlinear.ConstraintIndex,Float64}()
    ci_shift = Dict{MOI.Nonlinear.ConstraintIndex,Float64}()
    myrow_of_row = Vector{Int}(undef, length(rows))
    myrow = 0
    for (k, r) in enumerate(rows)
        r.is_ineq || continue
        myrow += 1
        myrow_of_row[k] = myrow
        if r.ci !== nothing
            ci_to_myrow[r.ci] = myrow
            ci_sign[r.ci] = r.sign
            ci_shift[r.ci] = r.shift
        end
    end
    nineq = myrow
    for (k, r) in enumerate(rows)
        r.is_ineq && continue
        myrow += 1
        myrow_of_row[k] = myrow
        if r.ci !== nothing
            ci_to_myrow[r.ci] = myrow
            ci_sign[r.ci] = r.sign
            ci_shift[r.ci] = r.shift
        end
    end
    neq = myrow - nineq
    ineq_linear = Bool[r.linear for r in rows if r.is_ineq]
    eq_linear = Bool[r.linear for r in rows if !r.is_ineq]
    ev_sign = zeros(length(ev.ordered_constraints))
    ev_shift = zeros(length(ev.ordered_constraints))
    myrow_of_evrow = zeros(Int, length(ev.ordered_constraints))
    for (k, ci) in enumerate(ev.ordered_constraints)
        myrow_of_evrow[k] = ci_to_myrow[ci]
        ev_sign[k] = ci_sign[ci]
        ev_shift[k] = ci_shift[ci]
    end
    jac_evrows = Int[]
    nl_jac_rows = Int[]
    nl_jac_cols = Int[]
    for (row, col) in MOI.jacobian_structure(ev)
        push!(nl_jac_rows, myrow_of_evrow[row])
        push!(nl_jac_cols, col)
        push!(jac_evrows, row)
    end
    nl_hess = Tuple{Int,Int}[(i, j) for (i, j) in MOI.hessian_lagrangian_structure(ev)]
    # Closed-form channel: affine and quadratic rows evaluated analytically.
    cl_rows = Int[]
    cl_sign = Float64[]
    cl_shift = Float64[]
    cl_q_id = Int[]
    cl_q_a = Int[]
    cl_q_b = Int[]
    cl_q_v = Float64[]
    cl_c_id = Int[]
    cl_c_col = Int[]
    cl_c_v = Float64[]
    cl_d = Float64[]
    for (k, r) in enumerate(rows)
        r.expr === nothing && continue
        id = length(cl_rows) + 1
        push!(cl_rows, myrow_of_row[k])
        push!(cl_sign, r.sign)
        push!(cl_shift, r.shift)
        if r.expr isa MOI.ScalarQuadraticFunction
            for t in r.expr.quadratic_terms
                push!(cl_q_id, id)
                push!(cl_q_a, positions[t.variable_1.value])
                push!(cl_q_b, positions[t.variable_2.value])
                push!(cl_q_v, t.coefficient)
            end
            for t in r.expr.affine_terms
                push!(cl_c_id, id)
                push!(cl_c_col, positions[t.variable.value])
                push!(cl_c_v, t.coefficient)
            end
            push!(cl_d, r.expr.constant)
        else
            for t in r.expr.terms
                push!(cl_c_id, id)
                push!(cl_c_col, positions[t.variable.value])
                push!(cl_c_v, t.coefficient)
            end
            push!(cl_d, r.expr.constant)
        end
    end
    # Merged Jacobian structure: the nonlinear entries (in evaluator order)
    # and one entry per (converted row, column) of the closed rows, sorted by
    # (row, column). Each closed entry records its evaluation recipe: a list
    # of (variable column, coefficient) products plus a constant.
    cl_entries = Tuple{Int,Int}[]
    cl_entry_terms = Vector{Tuple{Int,Float64}}[]
    cl_entry_const = Float64[]
    cl_entry_sign = Float64[]
    cl_pos_of = Dict{Tuple{Int,Int},Int}()
    for k in eachindex(cl_q_id)
        my = cl_rows[cl_q_id[k]]
        v = cl_q_v[k]
        a = cl_q_a[k]
        b = cl_q_b[k]
        # MOI stores quadratic terms as 0.5 * v * x_a^2 on the diagonal and
        # v * x_a * x_b off it, so the diagonal contributes a single
        # derivative entry v * x_a.
        pairs = a == b ? ((a, a),) : ((a, b), (b, a))
        for (c, src) in pairs
            key = (my, c)
            idx = get(cl_pos_of, key, 0)
            if idx == 0
                push!(cl_entries, key)
                push!(cl_entry_terms, Tuple{Int,Float64}[])
                push!(cl_entry_const, 0.0)
                push!(cl_entry_sign, cl_sign[cl_q_id[k]])
                idx = length(cl_entries)
                cl_pos_of[key] = idx
            end
            push!(cl_entry_terms[idx], (src, v))
        end
    end
    for k in eachindex(cl_c_id)
        key = (cl_rows[cl_c_id[k]], cl_c_col[k])
        idx = get(cl_pos_of, key, 0)
        if idx == 0
            push!(cl_entries, key)
            push!(cl_entry_terms, Tuple{Int,Float64}[])
            push!(cl_entry_const, 0.0)
            push!(cl_entry_sign, cl_sign[cl_c_id[k]])
            idx = length(cl_entries)
            cl_pos_of[key] = idx
        end
        cl_entry_const[idx] += cl_c_v[k]
    end
    all_jac = vcat(
        Tuple{Int,Int}[(nl_jac_rows[k], nl_jac_cols[k]) for k in eachindex(nl_jac_rows)],
        cl_entries,
    )
    p = sortperm(all_jac)
    jac_rows = [all_jac[k][1] for k in p]
    jac_cols = [all_jac[k][2] for k in p]
    invp = invperm(p)
    nl_jac_pos = invp[1:length(nl_jac_rows)]
    cl_jac_pos = invp[(length(nl_jac_rows) + 1):end]
    # Merged Hessian structure (lower triangle) over all channels: nonlinear
    # entries (in evaluator order), one entry per closed-form quadratic term,
    # and one per objective quadratic term.
    cl_hess_terms = Tuple{Int,Int}[]
    cl_hess_myrow = Int[]
    cl_hess_coef = Float64[]
    for k in eachindex(cl_q_id)
        a, b = cl_q_a[k], cl_q_b[k]
        push!(cl_hess_terms, (min(a, b), max(a, b)))
        push!(cl_hess_myrow, cl_rows[cl_q_id[k]])
        push!(cl_hess_coef, cl_q_v[k])
    end
    obj_q_a = Int[]
    obj_q_b = Int[]
    obj_q_v = Float64[]
    obj_c = zeros(nvars)
    obj_d = 0.0
    obj_hess_terms = Tuple{Int,Int}[]
    obj_hess_coef = Float64[]
    obj_closed_bool = obj_closed !== nothing
    if obj_closed isa MOI.ScalarQuadraticFunction
        for t in obj_closed.quadratic_terms
            a = positions[t.variable_1.value]
            b = positions[t.variable_2.value]
            push!(obj_q_a, a)
            push!(obj_q_b, b)
            push!(obj_q_v, t.coefficient)
            push!(obj_hess_terms, (min(a, b), max(a, b)))
            push!(obj_hess_coef, t.coefficient)
        end
        for t in obj_closed.affine_terms
            obj_c[positions[t.variable.value]] += t.coefficient
        end
        obj_d = obj_closed.constant
    elseif obj_closed isa MOI.ScalarAffineFunction
        for t in obj_closed.terms
            obj_c[positions[t.variable.value]] += t.coefficient
        end
        obj_d = obj_closed.constant
    end
    all_hess = vcat(nl_hess, cl_hess_terms, obj_hess_terms)
    hp = sortperm(all_hess)
    # Dedupe the merged structure: several channels can share a (row, column)
    # entry, and every channel's contribution accumulates into it.
    hess_rows = Int[]
    hess_cols = Int[]
    merged_pos_of_sorted = Vector{Int}(undef, length(all_hess))
    for (s, entry) in enumerate(all_hess[hp])
        if isempty(hess_rows) || entry != (hess_rows[end], hess_cols[end])
            push!(hess_rows, entry[1])
            push!(hess_cols, entry[2])
        end
        merged_pos_of_sorted[s] = length(hess_rows)
    end
    hinvp = invperm(hp)
    src_merged = [merged_pos_of_sorted[hinvp[k]] for k in eachindex(all_hess)]
    nl_hess_pos = src_merged[1:length(nl_hess)]
    cl_hess_pos = src_merged[(length(nl_hess) + 1):(length(nl_hess) + length(cl_hess_terms))]
    obj_hess_pos = src_merged[(length(nl_hess) + length(cl_hess_terms) + 1):end]
    return JuMPEvaluator(
        ev,
        vars,
        objective_sign,
        nineq,
        neq,
        jac_rows,
        jac_cols,
        jac_evrows,
        hess_rows,
        hess_cols,
        ev_sign,
        ev_shift,
        ineq_linear,
        eq_linear,
        myrow_of_evrow,
        zeros(length(ev.ordered_constraints)),
        zeros(length(jac_evrows)),
        zeros(length(nl_hess)),
        zeros(length(ev.ordered_constraints)),
        cl_rows,
        cl_sign,
        cl_shift,
        cl_q_id,
        cl_q_a,
        cl_q_b,
        cl_q_v,
        cl_c_id,
        cl_c_col,
        cl_c_v,
        cl_d,
        zeros(length(cl_rows)),
        nl_jac_pos,
        cl_jac_pos,
        cl_entry_terms,
        cl_entry_const,
        cl_entry_sign,
        nl_hess_pos,
        cl_hess_pos,
        cl_hess_myrow,
        cl_hess_coef,
        obj_hess_pos,
        obj_hess_coef,
        obj_closed_bool,
        obj_q_a,
        obj_q_b,
        obj_q_v,
        obj_c,
        obj_d,
    )
end

# Model-level callables: they take the model's structured variable container
# and evaluate the corresponding part of the single nonlinear model.

"""
    JuMPModelObjective(m::JuMPEvaluator, ks)

The objective of a JuMP-converted model: evaluates the converted objective
expression on the model's variable container through the shared
`JuMPEvaluator`.
"""
struct JuMPModelObjective{M,K} <: Function
    m::M
    ks::K
end
(f::JuMPModelObjective)(x) = JuMPModelObjectiveVector(f.m)(flatten(x, f.ks)[1])

"""
    JuMPModelIneq(m::JuMPEvaluator, ks)

The inequality constraints of a JuMP-converted model: a single vector-valued
function evaluating every converted inequality row (in the converted row
order) through the shared `JuMPEvaluator`.
"""
struct JuMPModelIneq{M,K} <: Function
    m::M
    ks::K
end
(f::JuMPModelIneq)(x) = JuMPModelIneqVector(f.m)(flatten(x, f.ks)[1])

"""
    JuMPModelEq(m::JuMPEvaluator, ks)

The equality constraints of a JuMP-converted model: a single vector-valued
function evaluating every converted equality row through the shared
`JuMPEvaluator`.
"""
struct JuMPModelEq{M,K} <: Function
    m::M
    ks::K
end
(f::JuMPModelEq)(x) = JuMPModelEqVector(f.m)(flatten(x, f.ks)[1])

# Vector-level callables: these are what reverse-mode backends encounter
# after the model variables are flattened, and each carries a ChainRules
# `rrule` providing the derivatives from the evaluator directly.

struct JuMPModelObjectiveVector{M} <: Function
    m::M
end
function (f::JuMPModelObjectiveVector)(x::AbstractVector)
    return f.m.objective_sign * jump_objective_value(f.m, x)
end
function ChainRulesCore.rrule(f::JuMPModelObjectiveVector, x::AbstractVector)
    y = f(x)
    return y, Δ -> (NoTangent(), jump_objective_gradient(f.m, x) .* _unthunk(Δ))
end

struct JuMPModelIneqVector{M} <: Function
    m::M
end
function (f::JuMPModelIneqVector)(x::AbstractVector)
    g = zeros(f.m.nineq + f.m.neq)
    jump_constraint_values!(f.m, g, x)
    return g[1:f.m.nineq]
end
function ChainRulesCore.rrule(f::JuMPModelIneqVector, x::AbstractVector)
    y = f(x)
    J = jump_jacobian(f.m, x)[1:f.m.nineq, :]
    return y, Δ -> (NoTangent(), J' * _unthunk(Δ))
end

struct JuMPModelEqVector{M} <: Function
    m::M
end
function (f::JuMPModelEqVector)(x::AbstractVector)
    g = zeros(f.m.nineq + f.m.neq)
    jump_constraint_values!(f.m, g, x)
    return g[(f.m.nineq + 1):end]
end
function ChainRulesCore.rrule(f::JuMPModelEqVector, x::AbstractVector)
    y = f(x)
    J = jump_jacobian(f.m, x)[(f.m.nineq + 1):end, :]
    return y, Δ -> (NoTangent(), J' * _unthunk(Δ))
end

"""
    jump_objective_value(m::JuMPEvaluator, x)

Returns the value of the converted objective at `x`: from the MathOptInterface
Nonlinear evaluator for nonlinear objectives, in closed form otherwise.
"""
function jump_objective_value(m::JuMPEvaluator, x)
    m.obj_closed || return MOI.eval_objective(m.ev, x)
    v = m.obj_d
    @inbounds for k in eachindex(m.obj_q_a)
        a = m.obj_q_a[k]
        b = m.obj_q_b[k]
        v += (a == b ? 0.5 * m.obj_q_v[k] * x[a] * x[a] : m.obj_q_v[k] * x[a] * x[b])
    end
    return v + dot(m.obj_c, x)
end

"""
    jump_objective_gradient(m::JuMPEvaluator, x)

Returns the gradient of the converted objective at `x`, computed by the
MathOptInterface Nonlinear evaluator.
"""
function jump_objective_gradient(m::JuMPEvaluator, x)
    g = zeros(length(x))
    jump_objective_gradient!(m, g, x)
    return g
end

function jump_objective_gradient!(m::JuMPEvaluator, g, x)
    if m.obj_closed
        copyto!(g, m.obj_c)
        @inbounds for k in eachindex(m.obj_q_a)
            a = m.obj_q_a[k]
            b = m.obj_q_b[k]
            v = m.obj_q_v[k]
            g[a] += v * x[b]
            a == b || (g[b] += v * x[a])
        end
    else
        MOI.eval_objective_gradient(m.ev, g, x)
    end
    m.objective_sign == 1.0 || (g .*= m.objective_sign)
    return g
end

"""
    jump_jacobian_structure(m::JuMPEvaluator)

Returns `(rows, columns)` of the converted model's constraint Jacobian, with
rows ordered inequality-first like the constraint values.
"""
jump_jacobian_structure(m::JuMPEvaluator) = (m.jac_rows, m.jac_cols)

"""
    jump_constraint_linearity(m::JuMPEvaluator)

Returns `(ineq_linear, eq_linear)`: per-row `Bool`s recording whether each
converted inequality and equality row is a linear expression.
"""
jump_constraint_linearity(m::JuMPEvaluator) = (m.ineq_linear, m.eq_linear)

"""
    jump_jacobian(m::JuMPEvaluator, x)

Returns the constraint Jacobian of the converted model at `x` as a sparse
matrix, computed by the MathOptInterface Nonlinear evaluator.
"""
function jump_jacobian(m::JuMPEvaluator, x)
    vals = zeros(length(m.jac_rows))
    jump_jacobian!(m, vals, x)
    return sparse(m.jac_rows, m.jac_cols, vals, m.nineq + m.neq, length(x))
end

function jump_jacobian!(m::JuMPEvaluator, vals, x)
    if !isempty(m.jac_evrows)
        MOI.eval_constraint_jacobian(m.ev, m.jac_buffer, x)
        @inbounds for (k, evrow) in enumerate(m.jac_evrows)
            vals[m.nl_jac_pos[k]] = m.ev_sign[evrow] * m.jac_buffer[k]
        end
    end
    @inbounds for k in eachindex(m.cl_jac_pos)
        acc = m.cl_jac_const[k]
        for (src, c) in m.cl_jac_terms[k]
            acc += c * x[src]
        end
        vals[m.cl_jac_pos[k]] = m.cl_jac_sign[k] * acc
    end
    return vals
end

"""
    jump_hessian_structure(m::JuMPEvaluator)

Returns `(rows, columns)` of the converted model's Lagrangian Hessian in
lower-triangular form.
"""
jump_hessian_structure(m::JuMPEvaluator) = (m.hess_rows, m.hess_cols)

function jump_hessian_lagrangian!(m::JuMPEvaluator, vals, x, obj_factor, lambda)
    fill!(vals, 0.0)
    if !isempty(m.myrow_of_evrow)
        μ = m.μ_buffer
        @inbounds for k in eachindex(m.myrow_of_evrow)
            μ[k] = m.ev_sign[k] * lambda[m.myrow_of_evrow[k]]
        end
        MOI.eval_hessian_lagrangian(
            m.ev, m.hess_buffer, x, obj_factor * m.objective_sign, μ
        )
        @inbounds for k in eachindex(m.nl_hess_pos)
            vals[m.nl_hess_pos[k]] += m.hess_buffer[k]
        end
    end
    @inbounds for k in eachindex(m.cl_hess_pos)
        vals[m.cl_hess_pos[k]] += lambda[m.cl_hess_myrow[k]] * m.cl_hess_coef[k]
    end
    if m.obj_closed
        @inbounds for k in eachindex(m.obj_hess_pos)
            vals[m.obj_hess_pos[k]] += obj_factor * m.objective_sign * m.obj_hess_coef[k]
        end
    end
    return vals
end

function jump_constraint_values!(m::JuMPEvaluator, g, x)
    if !isempty(m.myrow_of_evrow)
        MOI.eval_constraint(m.ev, m.g_buffer, x)
        @inbounds for k in eachindex(m.myrow_of_evrow)
            my = m.myrow_of_evrow[k]
            g[my] = m.ev_sign[k] * m.g_buffer[k] + m.ev_shift[k]
        end
    end
    if !isempty(m.cl_rows)
        fill!(m.cl_g_buffer, 0.0)
        @inbounds for k in eachindex(m.cl_q_id)
            id = m.cl_q_id[k]
            a = m.cl_q_a[k]
            b = m.cl_q_b[k]
            m.cl_g_buffer[id] += (
                a == b ? 0.5 * m.cl_q_v[k] * x[a] * x[a] : m.cl_q_v[k] * x[a] * x[b]
            )
        end
        @inbounds for k in eachindex(m.cl_c_id)
            m.cl_g_buffer[m.cl_c_id[k]] += m.cl_c_v[k] * x[m.cl_c_col[k]]
        end
        @inbounds for (id, my) in enumerate(m.cl_rows)
            g[my] = m.cl_sign[id] * (m.cl_g_buffer[id] + m.cl_d[id]) + m.cl_shift[id]
        end
    end
    return g
end

# The MathOptInterface evaluator cannot be traced by SparseConnectivityTracer
# (its internals branch on values); the converted model's derivatives come
# from the evaluator directly instead.

function sparse_jacobian(f::JuMPModelIneq, x, backend::AbstractADType=DEFAULT_AD_BACKEND[])
    return jump_jacobian(f.m, flatten(x, f.ks)[1])[1:f.m.nineq, :]
end

function sparse_jacobian(f::JuMPModelEq, x, backend::AbstractADType=DEFAULT_AD_BACKEND[])
    return jump_jacobian(f.m, flatten(x, f.ks)[1])[(f.m.nineq + 1):end, :]
end

function sparse_jacobian(
    c::VectorOfFunctions, x, backend::AbstractADType=DEFAULT_AD_BACKEND[]
)
    raws = map(c.fs) do f
        g = f.f isa FlattenedFunction ? getrawfunction(f.f) : f.f
        g = g isa FunctionWrapper ? getfunction(g) : g
        return g
    end
    if length(raws) == 1 && raws[1] isa JuMPModelIneq
        m = raws[1].m
        xv = x isa AbstractVector ? x : flatten(x, raws[1].ks)[1]
        return jump_jacobian(m, xv)[1:m.nineq, :]
    elseif length(raws) == 1 && raws[1] isa JuMPModelEq
        m = raws[1].m
        xv = x isa AbstractVector ? x : flatten(x, raws[1].ks)[1]
        return jump_jacobian(m, xv)[(m.nineq + 1):end, :]
    end
    return _sparse_jacobian_default(c, x, backend)
end

# The evaluator's internals cannot be differentiated by tracing backends, so
# the value-and-derivative family evaluates converted model functions
# directly at the flattened vector.

function value_gradient(
    f::FlattenedFunction{<:JuMPModelObjective},
    x::AbstractVector,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    jf = getrawfunction(f)
    m = jf.m
    return m.objective_sign * jump_objective_value(m, x), jump_objective_gradient(m, x)
end

function value_jacobian(
    f::FlattenedFunction{<:JuMPModelObjective},
    x::AbstractVector,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    jf = getrawfunction(f)
    m = jf.m
    y = m.objective_sign * jump_objective_value(m, x)
    return y, adjoint(jump_objective_gradient(m, x))
end

function value_jacobian(
    f::FlattenedFunction{<:FunctionWrapper{<:JuMPModelIneq}},
    x::AbstractVector,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    jf = getfunction(getrawfunction(f))
    m = jf.m
    g = zeros(m.nineq + m.neq)
    jump_constraint_values!(m, g, x)
    J = jump_jacobian(m, x)[1:m.nineq, :]
    return g[1:m.nineq], J
end

function value_jacobian(
    f::FlattenedFunction{<:FunctionWrapper{<:JuMPModelEq}},
    x::AbstractVector,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    jf = getfunction(getrawfunction(f))
    m = jf.m
    g = zeros(m.nineq + m.neq)
    jump_constraint_values!(m, g, x)
    J = jump_jacobian(m, x)[(m.nineq + 1):end, :]
    return g[(m.nineq + 1):end], J
end
