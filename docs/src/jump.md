# JuMP models

JuMP.jl has an excellent API for defining variables and constraints. In
Nonconvex, you can start with a JuMP model, define variables and constraints
using JuMP's API, then convert it to a `DictModel`:

```julia
jump_model = JuMP.Model()
@variable jump_model 0 <= x[i=1:3] <= 1
@constraint jump_model sum(x) <= 1
model = DictModel(jump_model)
```

The objective can also be defined either using JuMP or Nonconvex. Once you
convert the JuMP model to a Nonconvex model, you can go ahead and define more
variables and constraints and/or set the objective in Nonconvex.

> JuMP is an optional dependency of NonconvexCore: the conversion methods
> are provided by the `NonconvexCoreJuMPExt` package extension. To use them,
> load JuMP first (e.g. `using JuMP`) so the extension activates.

## Affine and quadratic models

When the JuMP model contains no nonlinear expressions, the conversion uses
the exact affine and quadratic forms: the objective becomes a quadratic form
`c'x + x'Qx` and the constraints become matrix-vector products with their
bound offsets. Variable bounds (including `Interval` sets), binary and
integer flags, and start values are all carried over.

## Nonlinear models

When the JuMP model contains any nonlinear expression (from JuMP's nonlinear
interface, e.g. `sin`, `cos`), the objective and every affine, quadratic and
nonlinear constraint are added to a single `MOI.Nonlinear.Model`, and the
converted model is evaluated by one shared `MOI.Nonlinear.Evaluator`:

- the objective value and gradient come from `MOI.eval_objective` and
  `MOI.eval_objective_gradient`;
- all constraint values come from one `MOI.eval_constraint` pass, so shared
  subexpressions are evaluated once per iterate;
- the sparse constraint Jacobian comes from one `MOI.eval_constraint_jacobian`
  pass;
- the Lagrangian Hessian `obj_factor * ∇²f + Σᵢ λᵢ ∇²cᵢ` comes from one
  `MOI.eval_hessian_lagrangian` pass.

Solvers that support it (e.g. NonconvexIpopt) detect converted models and
drive the evaluator directly, bypassing the automatic differentiation
machinery entirely.

Nonconvex inequality constraints are feasible when their value is `<= 0`, so
the converted rows carry the sign and offset of the original set:
`GreaterThan(l)` becomes `l - f`, `LessThan(u)` becomes `f - u`, `EqualTo(v)`
becomes `f - v`, and each `Interval(l, u)` becomes the two rows `l - f` and
`f - u` evaluated by a single call.

## Evaluating single expressions

[`JuMPSymbolicFunction`](@ref) wraps one MathOptInterface scalar function
(any of `MOI.ScalarNonlinearFunction`, `MOI.ScalarQuadraticFunction` or
`MOI.ScalarAffineFunction`) together with the ordered variables it depends
on, and evaluates its value, gradient and Hessian with the
`MOI.Nonlinear` evaluator:

```julia
expr = MOI.ScalarNonlinearFunction(:sin, Any[MOI.VariableIndex(1)])
f = JuMPSymbolicFunction(expr, [MOI.VariableIndex(1)])
f([0.5])                  # value
jump_symbolic_gradient(f, [0.5])
jump_symbolic_hessian(f, [0.5])
```

A `ChainRulesCore` `rrule` is defined, so the function differentiates
correctly under any ChainRules-based reverse-mode backend.
