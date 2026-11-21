# NEWS

## 2.2.9

- `CustomHessianFunction` now supports vector-valued functions with `hvp = true`: `h(x, v)` returns the per-component Hessian-vector products and the weighted Lagrangian Hessian `Σ μᵢ∇²fᵢ(x)` is assembled from them, so second-order solvers get exact constraint curvature from a vector-valued block (closes #67). HVP-operator gradients also flow through `CustomGradFunction`'s rrule and frule.
## 2.2.8

- JuMP-converted model functions reject `ForwardDiff.Dual` inputs with an informative `ArgumentError` (nested AD through the evaluator cannot work; use the evaluator's own derivatives instead).
- The affine/quadratic JuMP conversion preserves constant terms: `get_constraint_info` uses `bound - func.constant` and `get_objective_info` keeps the objective constant with the correct Maximization sign (previously a converted quadratic objective could report e.g. -4.5 instead of 0.5).
## 2.2.7

- Exact JuMP evaluator backends in the ADNLPModels compatibility
  extension: `JuMPJacBackend`, `JuMPHvBackend` and `JuMPHessBackend` read
  derivatives straight from the `JuMPEvaluator` (exact and Float64-safe),
  and the `Zygote*` backend constructors dispatch to them whenever
  `jump_evaluator` is passed — algorithm packages get exact JuMP-converted
  derivatives without owning any AD code.
- `value_jacobian` returns the well-defined zero row for constant
  functions instead of erroring on a `nothing` gradient.
- `NLPModels` added as a dependency of the ADNLPModels extension.

## 2.2.4

- Added docstrings for the modelling API (`addvar!`, the constraint adders, `getmin`/`getmax`/`getobjective`, `set_objective!`, `setmin!`/`setmax!`/`setinteger!`, `diffy` and the `prepare_*` functions), visible in the REPL help and the API reference.
