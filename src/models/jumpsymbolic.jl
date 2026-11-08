# Functions whose derivatives come from MathOptInterface's Nonlinear
# evaluator (the stack JuMP uses for its new nonlinear interface), used for
# JuMP models converted into NonconvexCore models.

_unthunk(x) = x
_unthunk(x::ChainRulesCore.AbstractThunk) = ChainRulesCore.unthunk(x)

"""
    JuMPSymbolicFunction(expr::MOI.ScalarNonlinearFunction, vars::Vector{MOI.VariableIndex})

Returns a callable that evaluates `expr` at a flat vector `x` (ordered like
`vars`) with value, gradient and Hessian computed by MathOptInterface's
`Nonlinear` sparse reverse-mode evaluator. Reverse-mode differentiation of
the callable uses a ChainRules `rrule`; the gradient and Hessian are also
available via [`jump_symbolic_gradient`](@ref) and
[`jump_symbolic_hessian`](@ref).
"""
struct JuMPSymbolicFunction{E} <: Function
    evaluator::E
end

function JuMPSymbolicFunction(expr::MOI.ScalarNonlinearFunction, vars::Vector{MOI.VariableIndex})
    model = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(model, expr)
    evaluator = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    MOI.initialize(evaluator, [:Grad, :Hess])
    return JuMPSymbolicFunction(evaluator)
end

(f::JuMPSymbolicFunction)(x) = MOI.eval_objective(f.evaluator, x)

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
    structure = MOI.hessian_objective_structure(f.evaluator)
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
