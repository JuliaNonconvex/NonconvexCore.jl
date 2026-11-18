# NEWS

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
