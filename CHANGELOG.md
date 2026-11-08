# Changelog

All notable changes to `NonconvexCore.jl` are recorded here.
The format follows the recommendation from JuliaRegistrator:
a brief summary, then categorized lists of breaking changes,
new features, bug fixes, and maintenance items.

## v2.0.0 (2026-11-08)

### Breaking changes
- The capabilities system is now **strict by default**: every model feature
  present (equality/inequality/semidefinite constraints, finite variable
  bounds, integer variables, infinite variable bounds) must be declared
  supported by the solver via `solver_capabilities`, otherwise `optimize`
  throws an informative `ArgumentError`. Solvers that previously ignored
  constraints silently (or with a warning) must now declare their
  capabilities. The semidefinite-constraints warning is removed: undeclared
  solvers error instead.
- `ConstraintCapabilities` gains an `infinite_bounds` field: solvers declare
  whether they support variables with infinite (unbounded) bounds, and
  `checksubproblem!` validates models with such variables against the
  declaration.

### New features
- Add the strict capabilities system: `ConstraintCapabilities`,
  `solver_capabilities` and `checksubproblem!` plus the
  `merge_capabilities` helper. Solvers declaring capabilities are validated
  by the generic `optimize` against the features present in the model
  (equality/inequality/semidefinite constraints, variable bounds, integer
  variables and infinite bounds), with an informative `ArgumentError` for
  unsupported or undeclared features.
- `value_gradient`, `value_jacobian`, `value_jacobian_transpose` and the new
  `value_hessian` accept an `ADTypes.AbstractADType` backend argument and
  default to the package-wide backend, initially `AutoZygote`, settable with
  `set_ad_backend!`.
- `sparse_jacobian` and `sparse_hessian` detect the structural sparsity
  pattern with SparseConnectivityTracer, color it with SparseMatrixColorings
  and evaluate the compressed derivatives through DifferentiationInterface
  with the selected backend; both return a `SparseMatrixCSC` for any input
  function. If the sparse pipeline fails for a function, the values are
  assembled from per-function derivatives at the detected pattern positions.
- The MOI evaluator detects the sparse Jacobian and Hessian structures once
  at the initial point and evaluates derivative values with the selected
  backend, extracting them at the detected positions.

### Bug fixes
- Reverse-mode differentiation of a `VectorOfFunctions` with more than one
  function used to corrupt the seed of every function (Zygote's `vcat`
  pullback over scalar outputs), producing wrong whole-collection Jacobians
  and Hessians. The collection now defines a pullback assembled from
  per-function pullbacks, and the Lagrangian in the MOI evaluator
  accumulates one term per constraint.

## v1.6.0 (2026-08-11)

### New features
- Accept `DifferentiableFlatten v0.2` (the release that introduced the
  `julia = "1.10"` floor).

### Maintenance
- Raise the minimum Julia version to `1.10`.
  ([#24](https://github.com/JuliaNonconvex/NonconvexCore.jl/pull/24)).
- Standardize `[compat]` entries; stdlib bounds normalized to `"1"`.
  Sorted multi-version compat entries
  (e.g., dropped the unreachable `ChainRulesCore = 0.10, 0.9, 1` → `0.10, 1`).
  ([#24](https://github.com/JuliaNonconvex/NonconvexCore.jl/pull/24)).
- Update the GitHub handle in the convergence-plots docstring URL from
  `mohamed82008` to `mohdibntarek`
  ([#25](https://github.com/JuliaNonconvex/NonconvexCore.jl/pull/25)).

## v1.5.1 (2024-11-22)

### Bug fixes
- Allow `Zygote 0.7` and bump `ChainRulesCore` compat.

## v1.5.0 (2024-02-...)

For earlier changes see the git history.
