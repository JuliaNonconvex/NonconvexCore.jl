# Changelog

All notable changes to `NonconvexCore.jl` are recorded here.
The format follows the recommendation from JuliaRegistrator:
a brief summary, then categorized lists of breaking changes,
new features, bug fixes, and maintenance items.

## v2.0.0 (2026-11-08)

### Breaking changes
- The capabilities system is now **strict by default**: every model feature
  present (equality/inequality/semidefinite constraints, finite variable
  bounds, integer variables) must be declared supported by the solver via
  `solver_capabilities`, otherwise `optimize` throws an informative
  `ArgumentError`. Solvers that previously ignored constraints silently (or
  with a warning) must now declare their capabilities. The
  semidefinite-constraints warning is removed: undeclared solvers error
  instead.

### New features
- Add the strict capabilities system: `ConstraintCapabilities`,
  `solver_capabilities` and `checksubproblem!` plus the
  `merge_capabilities` helper. Solvers declaring capabilities are validated
  by the generic `optimize` against the features present in the model
  (equality/inequality/semidefinite constraints, variable bounds and integer
  variables), with an informative `ArgumentError` for unsupported or
  undeclared features.

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
