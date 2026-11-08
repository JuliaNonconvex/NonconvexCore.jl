# JuMP-model conversion API. The JuMP-specific methods live in the
# NonconvexCoreJuMPExt package extension, because JuMP is an optional
# dependency of NonconvexCore. These stubs declare the shared names.

"""
    get_variable_info(model::JuMP.Model)

Returns the lower bounds, upper bounds, initial points and integer flags of
the variables of a JuMP model as `OrderedDict`s, in the structure NonconvexCore
models use.
"""
function get_variable_info end

"""
    _ordered_jump_variables(model::JuMP.Model, ks)

Returns the JuMP model's variables ordered like the flattened model vector
with keys `ks`.
"""
function _ordered_jump_variables end

"""
    _nonlinear_constraints(model::JuMP.Model)

Returns the model's nonlinear (and quadratic and affine, when a nonlinear
expression is present) constraints as `(expr, set)` pairs with the
expressions converted to MathOptInterface scalar functions.
"""
function _nonlinear_constraints end

"""
    get_jump_problem(model::VecModel, x0 = getinit(model); optimizer, first_order, backend, kwargs...)

Builds a JuMP model wrapping a NonconvexCore model through a custom
`MOI.AbstractNLPEvaluator`, for solvers that interface through JuMP.
"""
function get_jump_problem end
