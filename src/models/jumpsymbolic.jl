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

function JuMPSymbolicFunction(expr::MOI.AbstractScalarFunction, vars::Vector{MOI.VariableIndex})
    model = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(model, expr)
    evaluator = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
    MOI.initialize(evaluator, [:Grad, :Hess])
    return JuMPSymbolicFunction(evaluator, expr, vars)
end

(f::JuMPSymbolicFunction)(x) = MOI.eval_objective(f.evaluator, x)

# The variable leaves of the expression, in order of appearance.
function _variable_leaves(expr::MOI.ScalarNonlinearFunction, out = Int[])
    for arg in expr.args
        if arg isa MOI.VariableIndex
            push!(out, arg.value)
        elseif arg isa MOI.ScalarNonlinearFunction
            _variable_leaves(arg, out)
        elseif arg isa MOI.ScalarQuadraticFunction
            for term in arg.quadratic_terms
                push!(out, term.variable_1.value)
                push!(out, term.variable_2.value)
            end
            for term in arg.affine_terms
                push!(out, term.variable.value)
            end
        elseif arg isa MOI.ScalarAffineFunction
            for term in arg.terms
                push!(out, term.variable.value)
            end
        end
    end
    return out
end

"""
    jump_symbolic_jacobian_sparsity(f::JuMPSymbolicFunction, n)

Returns the column indices of the variables that the wrapped expression
depends on, i.e. the sparsity pattern of its gradient with respect to a
vector of length `n`.
"""
function jump_symbolic_jacobian_sparsity(f::JuMPSymbolicFunction, n)
    positions = Dict(v.value => k for (k, v) in enumerate(f.vars))
    return Int[positions[v] for v in _variable_leaves(f.expr)]
end

"""
    jump_symbolic_hessian_sparsity(f::JuMPSymbolicFunction, n)

Returns the sparse Hessian structure `(row, column)` of the wrapped
expression with respect to a vector of length `n`.
"""
function jump_symbolic_hessian_sparsity(f::JuMPSymbolicFunction, n)
    positions = Dict(v.value => k for (k, v) in enumerate(f.vars))
    return [(positions[i], positions[j]) for (i, j) in MOI.hessian_objective_structure(f.evaluator)]
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

# A callable that evaluates a JuMP-sourced expression through its
# JuMPSymbolicFunction on the flattened model vector. The wrapped function is
# exposed so that sparse differentiation can use the expression graph's
# sparsity directly.
struct JumpConvertedFunction{F,U} <: Function
    f::F
    ks::U
end
(g::JumpConvertedFunction)(x) = g.f(flatten(x, g.ks)[1])

"""
    jump_symbolic_function(f::JumpConvertedFunction)

Returns the [`JuMPSymbolicFunction`](@ref) wrapped by `f`.
"""
jump_symbolic_function(g::JumpConvertedFunction) = g.f

# A bound that negates the expression: `lower - f(x)` for `f >= lower`. JuMP
# may or may not have folded the bound into the expression (its nonlinear
# interface normalizes to a zero set, affine constraints do not), so the raw
# bound is carried explicitly.
struct NegatedJumpConvertedFunction{F,U,L} <: Function
    f::F
    ks::U
    lower::L
end
(g::NegatedJumpConvertedFunction)(x) = g.lower - g.f(x)
jump_symbolic_function(g::NegatedJumpConvertedFunction) = jump_symbolic_function(g.f)

# A bound that shifts the expression: `f(x) - shift` for `f <= shift` or
# `f == shift`. The shift does not affect the derivatives.
struct ShiftedJumpConvertedFunction{F,U,S} <: Function
    f::F
    ks::U
    shift::S
end
(g::ShiftedJumpConvertedFunction)(x) = g.f(x) - g.shift
jump_symbolic_function(g::ShiftedJumpConvertedFunction) = jump_symbolic_function(g.f)

# An interval bound `lb <= f <= ub`: the two Nonconvex inequality values are
# `[lb - f(x), f(x) - ub]`, both feasible when <= 0, both from a single
# evaluation of the wrapped expression.
struct IntervalJumpConvertedFunction{F,U,L,S} <: Function
    f::F
    ks::U
    lower::L
    upper::S
end
function (g::IntervalJumpConvertedFunction)(x)
    v = g.f(x)
    return [g.lower - v, v - g.upper]
end
jump_symbolic_function(g::IntervalJumpConvertedFunction) = jump_symbolic_function(g.f)

function ChainRulesCore.rrule(f::JuMPSymbolicFunction, x)
    y = f(x)
    return y, Δ -> (NoTangent(), jump_symbolic_gradient(f, x) * _unthunk(Δ))
end

# JuMP-sourced functions: the expression graph provides the sparsity pattern
# directly, and the MathOptInterface evaluator cannot be traced by
# SparseConnectivityTracer (its internals branch on values).
function sparse_jacobian(f::JumpConvertedFunction, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    J = Matrix(value_jacobian(f, x, backend)[2])
    cols = jump_symbolic_jacobian_sparsity(jump_symbolic_function(f), length(x))
    pattern = sparse(1, cols, ones(length(cols)), 1, length(x))
    return J .* pattern
end

for W in (:NegatedJumpConvertedFunction, :ShiftedJumpConvertedFunction, :IntervalJumpConvertedFunction)
    @eval function sparse_jacobian(f::$W, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
        J = Matrix(value_jacobian(f, x, backend)[2])
        cols = jump_symbolic_jacobian_sparsity(jump_symbolic_function(f), length(x))
        rows = repeat(1:size(J, 1); inner = length(cols))
        colsrep = repeat(cols; outer = size(J, 1))
        pattern = sparse(rows, colsrep, ones(length(rows)), size(J, 1), length(x))
        return J .* pattern
    end
end

function sparse_jacobian(c::VectorOfFunctions, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    raws = map(c.fs) do f
        g = f.f isa FlattenedFunction ? getrawfunction(f.f) : f.f
        g = g isa FunctionWrapper ? getfunction(g) : g
        g
    end
    if !isempty(raws) && all(f -> f isa JumpConvertedFunction || f isa NegatedJumpConvertedFunction || f isa ShiftedJumpConvertedFunction || f isa IntervalJumpConvertedFunction, raws)
        I = Int[]
        Js = Int[]
        V = Float64[]
        rowout = 0
        for (k, f) in enumerate(raws)
            Jb = Matrix(value_jacobian(c.fs[k], x, backend)[2])
            cols = jump_symbolic_jacobian_sparsity(jump_symbolic_function(f), length(x))
            for lr in 1:size(Jb, 1)
                rowout += 1
                for cidx in cols
                    push!(I, rowout)
                    push!(Js, cidx)
                    push!(V, Jb[lr, cidx])
                end
            end
        end
        return sparse(I, Js, V, rowout, length(x))
    end
    return _sparse_jacobian_default(c, x, backend)
end
