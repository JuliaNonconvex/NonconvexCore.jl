# Changelog

All notable changes to `NonconvexCore.jl` are recorded here.
The format follows the recommendation from JuliaRegistrator:
a brief summary, then categorized lists of breaking changes,
new features, bug fixes, and maintenance items.

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
