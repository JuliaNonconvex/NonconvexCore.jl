# NonconvexCore

## Description

`NonconvexCore.jl` is the modelling and automatic differentiation foundation
of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem: every
solver package — Ipopt, NLopt, MMA, Percival, Metaheuristics, the
semidefinite and augmented Lagrangian methods, the Bayesian optimizers and
the reliability-analysis tools — builds on the same model types, constraint
machinery, derivative rules and solver interface.

Models are declared with [`Model`](@ref) or [`DictModel`](@ref): variables
with box and integrality bounds, an objective and any mix of equality,
inequality, interval and semidefinite constraints. As of NonconvexCore 2.3
there is also a unified function-in-set framework — [`EqualTo`](@ref),
[`LessThan`](@ref), [`GreaterThan`](@ref), [`Interval`](@ref),
[`PSDCone`](@ref), [`ChanceSet`](@ref) and [`SetProduct`](@ref) — with
[`add_set_constraint!`](@ref), [`add_set_variable!`](@ref) and
[`add_chance_constraint!`](@ref), the same interface consumed by the
augmented Lagrangian (NonconvexAugLagLab) and semidefinite
(NonconvexSemidefinite) meta-algorithms.

Gradients, Jacobians and Hessians come from the Zygote and ForwardDiff
backends behind a common ADTypes interface: [`set_ad_backend!`](@ref) and
[`get_ad_backend`](@ref) select the package-wide backend,
[`prepare_gradient`](@ref) prepares derivative evaluations explicitly,
[`diffy`](@ref NonconvexCore.diffy) attaches custom derivatives to plain
functions, [`LazyJacobian`](@ref) defers the evaluation of expensive
Jacobians and [`TraceFunction`](@ref) records function-call histories.
Symbolic models can be differentiated with Symbolics ([`symbolify`](@ref))
or converted from a JuMP model (`DictModel(jump_model)`, see
[JuMP models](jump.md)).

Every solver advertises which model features it supports through the solver
capability system, and declares the convergence criteria it understands;
[`getmin`](@ref), [`getmax`](@ref), [`getobjective`](@ref) and the other
model queries give solver-independent access to a model's state, and
[`set_violation`](@ref) measures the violation of a function-in-set
constraint.

See [Models](models.md) for model types, variables and constraints,
[JuMP models](jump.md) for the JuMP conversion path, [AD backends](derivatives/backends.md)
and [Sparse Jacobians and Hessians](derivatives/sparse.md) for the
derivative machinery, [Gradients, Jacobians and Hessians](gradients/gradients.md)
for user-defined derivatives, ChainRules in ForwardDiff and gradient-call
history, [Function modifiers](modifiers/custom.md) for `diffy`, lazy
Jacobians and call tracing, [Symbolic differentiation](symbolify.md) for the
Symbolics interface, [Solver capabilities](capabilities.md) for the
capability system, [Convergence criteria](criteria.md) for the criteria
interface, [Model queries](queries.md) for the query API,
[Optimization result](result.md) for the fields of the returned result and
[Docstrings](docstrings.md) for the full API reference.
