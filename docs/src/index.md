# NonconvexCore

`NonconvexCore` is the modelling and automatic differentiation core of the
Nonconvex ecosystem. It defines the optimization model types, the
objective and constraint functions, the differentiation machinery shared by
all the Nonconvex solver packages, and the solver capability system.

This documentation is a fragment of the [Nonconvex documentation](https://julianonconvex.github.io/Nonconvex.jl/)
and is also built standalone.

## Models

Two model types are provided:

- [`Model`](@ref): variables are stacked in one (flattened) vector.
- [`DictModel`](@ref): variables are named (a `Symbol` or `String` per
  variable) and collected in an `OrderedDict`.

Both support arbitrary variable structures (scalars, arrays, dictionaries,
structs, tuples, named tuples and nested combinations thereof), which are
flattened to a vector transparently by the differentiation machinery. See
[Models](models.md).

## Objectives and constraints

An objective function can be set with [`set_objective!`](@ref), and
inequality, equality and semidefinite constraints can be added with
[`add_ineq_constraint!`](@ref), [`add_eq_constraint!`](@ref) and
[`add_sd_constraint!`](@ref). Inequality constraints follow the convention
`f(x) <= 0` at feasibility.

## JuMP models

JuMP models can be converted to `DictModel`s with
`DictModel(::JuMP.Model)` when JuMP is loaded. Models with nonlinear
expressions are evaluated through one MathOptInterface `Nonlinear`
evaluator, so the values, sparse Jacobian and Lagrangian Hessian of the whole
model come from single evaluator passes. See [JuMP models](jump.md).

## Derivatives

Derivatives are computed through the
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
backend system. The package-wide default backend is `AutoZygote()`, and it
can be changed with [`set_ad_backend!`](@ref). See
[AD backends](derivatives/backends.md).

Structurally sparse Jacobians and Hessians can be computed with
[`sparse_jacobian`](@ref NonconvexCore.sparse_jacobian) and
[`sparse_hessian`](@ref NonconvexCore.sparse_hessian), which detect the
sparsity with SparseConnectivityTracer and color it with
SparseMatrixColorings. See [Sparse Jacobians and Hessians](derivatives/sparse.md).

## Function modifiers

Function modifiers customize how a specific function is differentiated,
without affecting other functions in the model:

- [`CustomGradFunction`](@ref) and [`CustomHessianFunction`](@ref) use
  user-supplied gradients, Jacobians and Hessians.
- [`LazyJacobian`](@ref) supplies matrix-free Jacobian-vector and
  vector-Jacobian products.
- [`TraceFunction`](@ref) records where a function is evaluated.
- [`ImplicitFunction`](@ref) differentiates implicit functions.
- [`symbolify`](@ref) computes derivatives symbolically with Symbolics.

See [Function modifiers](modifiers/custom.md), [Implicit functions](implicit.md)
and [Symbolic differentiation](symbolify.md).

## Solver capabilities

[`ConstraintCapabilities`](@ref) describes the constraint types and bounds a
solver supports, and [`checksubproblem!`](@ref) validates a model against a
solver's capabilities before optimization. See
[Solver capabilities](capabilities.md).

## Package extensions

NonconvexCore uses package extensions for its optional integrations:

- `NonconvexCoreJuMPExt`: the JuMP model conversion, loaded when JuMP is
  present in the environment.
- `NonconvexCoreSymbolicsExt`: [`symbolify`](@ref), loaded when Symbolics is
  present in the environment.
