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
    JuMPEvaluator

A single MathOptInterface `Nonlinear` evaluator over an entire JuMP model
converted into a NonconvexCore model. The objective and every constraint are
expressions of one `MOI.Nonlinear.Model`, so shared subexpressions are
evaluated once per pass, and the whole Jacobian and Lagrangian Hessian come
from single evaluator calls.

The constraints are keyed by their `ConstraintIndex`, because neither the
JuMP constraint iteration order nor the evaluator row order is meaningful;
the evaluator's own row order is recovered from `ordered_constraints`.
Converted rows are exposed inequality-first (sorted by constraint index),
then equalities (also sorted by constraint index). Inequality row `k` has
value `sign * f_k(x) + shift` (Nonconvex convention: feasible when <= 0);
equality row `k` has value `f_k(x) + shift` (feasible when 0). The objective
is `objective_sign * f_obj(x)`.
"""
mutable struct JuMPEvaluator{E,V}
    ev::E
    vars::V
    objective_sign::Float64
    nineq::Int
    neq::Int
    jac_rows::Vector{Int}      # converted rows, aligned with the jacobian values
    jac_cols::Vector{Int}
    jac_evrows::Vector{Int}    # evaluator rows, same alignment
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    ev_sign::Vector{Float64}   # per evaluator row
    ev_shift::Vector{Float64}  # per evaluator row
    ineq_linear::Vector{Bool}  # per converted row
    eq_linear::Vector{Bool}    # per converted row
    myrow_of_evrow::Vector{Int}
    g_buffer::Vector{Float64}  # constraint values in evaluator row order
    jac_buffer::Vector{Float64}# jacobian values in evaluator structure order
    μ_buffer::Vector{Float64}  # Lagrangian multipliers in evaluator row order
end

function JuMPEvaluator(
    model::MOI.Nonlinear.Model,
    vars::Vector{MOI.VariableIndex},
    objective_sign::Float64,
    ineq_cis::Vector{MOI.Nonlinear.ConstraintIndex},
    ineq_sign::Vector{Float64},
    ineq_shift::Vector{Float64},
    eq_cis::Vector{MOI.Nonlinear.ConstraintIndex},
    eq_shift::Vector{Float64},
    ineq_linear::Vector{Bool},
    eq_linear::Vector{Bool},
)
    ev = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    features = [:Grad, :Hess]
    isempty(ineq_cis) && isempty(eq_cis) || push!(features, :Jac)
    MOI.initialize(ev, features)
    nineq = length(ineq_cis)
    neq = length(eq_cis)
    m = nineq + neq
    # Converted rows: inequalities first (sorted by constraint index for a
    # deterministic order), then equalities.
    perm = sortperm([c.value for c in ineq_cis])
    ineq_cis = ineq_cis[perm]
    ineq_sign = ineq_sign[perm]
    ineq_shift = ineq_shift[perm]
    ineq_linear = ineq_linear[perm]
    perm = sortperm([c.value for c in eq_cis])
    eq_cis = eq_cis[perm]
    eq_shift = eq_shift[perm]
    eq_linear = eq_linear[perm]
    ci_to_myrow = Dict{MOI.Nonlinear.ConstraintIndex,Int}()
    ci_sign = Dict{MOI.Nonlinear.ConstraintIndex,Float64}()
    ci_shift = Dict{MOI.Nonlinear.ConstraintIndex,Float64}()
    for (k, ci) in enumerate(ineq_cis)
        ci_to_myrow[ci] = k
        ci_sign[ci] = ineq_sign[k]
        ci_shift[ci] = ineq_shift[k]
    end
    for (k, ci) in enumerate(eq_cis)
        ci_to_myrow[ci] = nineq + k
        ci_sign[ci] = 1.0
        ci_shift[ci] = eq_shift[k]
    end
    ev_sign = zeros(m)
    ev_shift = zeros(m)
    myrow_of_evrow = zeros(Int, m)
    for (k, ci) in enumerate(ev.ordered_constraints)
        myrow_of_evrow[k] = ci_to_myrow[ci]
        ev_sign[k] = ci_sign[ci]
        ev_shift[k] = ci_shift[ci]
    end
    jac_rows = Int[]
    jac_cols = Int[]
    jac_evrows = Int[]
    for (row, col) in MOI.jacobian_structure(ev)
        push!(jac_rows, myrow_of_evrow[row])
        push!(jac_cols, col)
        push!(jac_evrows, row)
    end
    hess_rows = Int[]
    hess_cols = Int[]
    for (i, j) in MOI.hessian_lagrangian_structure(ev)
        push!(hess_rows, i)
        push!(hess_cols, j)
    end
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
        zeros(m),
        zeros(length(jac_rows)),
        zeros(m),
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
    return f.m.objective_sign * MOI.eval_objective(f.m.ev, x)
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
    MOI.eval_objective_gradient(m.ev, g, x)
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
    MOI.eval_constraint_jacobian(m.ev, m.jac_buffer, x)
    for (k, evrow) in enumerate(m.jac_evrows)
        vals[k] = m.ev_sign[evrow] * m.jac_buffer[k]
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
    μ = m.μ_buffer
    for k in eachindex(m.myrow_of_evrow)
        μ[k] = m.ev_sign[k] * lambda[m.myrow_of_evrow[k]]
    end
    MOI.eval_hessian_lagrangian(m.ev, vals, x, obj_factor * m.objective_sign, μ)
    return vals
end

function jump_constraint_values!(m::JuMPEvaluator, g, x)
    MOI.eval_constraint(m.ev, m.g_buffer, x)
    for k in eachindex(m.myrow_of_evrow)
        my = m.myrow_of_evrow[k]
        g[my] = m.ev_sign[k] * m.g_buffer[k] + m.ev_shift[k]
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
    return m.objective_sign * MOI.eval_objective(m.ev, x), jump_objective_gradient(m, x)
end

function value_jacobian(
    f::FlattenedFunction{<:JuMPModelObjective},
    x::AbstractVector,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    jf = getrawfunction(f)
    m = jf.m
    y = m.objective_sign * MOI.eval_objective(m.ev, x)
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
