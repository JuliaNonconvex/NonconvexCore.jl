# NEWS

## 2.3.0

- Function-in-set constraints and variable-in-set blocks, the
  MathOptInterface-style modeling core: `FunctionInSet(f, set)` with the
  sets `EqualTo`, `LessThan`, `GreaterThan`, `Interval`, `Nonnegatives`,
  `Nonpositives`, `Zeros`, `Reals`, `Simplex`, `ChanceSet` and
  `SetProduct` — a `SetProduct` routes one bundle evaluation into several
  constraints of different kinds (scalar, vector and matrix-valued), so an
  expensive computation (e.g. a finite-element solve) feeds them all at
  once. `add_set_constraint!` and `add_chance_constraint!` register them;
  `add_set_variable!(m, set)` registers variable blocks (a `Simplex` block,
  a probability-simplex variable) whose cone parts become synthetic set
  constraints at model conversion.
- The classical constraint syntax is a view of the sets:
  `add_ineq_constraint!(m, f, s)` is `f ∈ InequalitySet(s)` and
  `add_eq_constraint!` is `f ∈ EqualitySet(s)`; `setof`/`tosetconstraint`
  convert classical constraints to the set form (the migration path to the
  unified function-in-set model of NonconvexCore 3). `addvar!` registers
  its block as an `Interval` set variable.
- Granular solver capabilities: `ConstraintCapabilities` now splits
  function-in-set support per set kind (`sets_eq`, `sets_ineq`, `sets_sd`,
  `sets_simplex`, `sets_product`, `sets_chance`, `variable_sets_sd`,
  `variable_sets_simplex`); `checksubproblem!` rejects models with
  unsupported set features with an informative error. The semidefinite cone
  kind (`sets_sd`, `:sd`) is provided by NonconvexSemidefinite.jl, which
  owns `PSDCone` and specializes Core's extension hooks (`sd_setof`, the
  set-kind, validation and variable-block hooks).
- Domain transforms: `nullspace_transform` and `unconstrained_transform`
  return `(model, x0, mapper)` and differentiate through the mapping with
  analytic ChainRules rrules (TransformVariables + Zygote), including a
  Stan-style simplex.
- Epigraph reformulation: `add_epigraph_variable!` and `epigraph!`.
- First-class linear constraints: `LinearFunction` and
  `add_linear_constraint!`.
- Shared multi-output evaluation: `SharedEval`, a cached single evaluation
  with exact gradient replay.
- The chordal-decomposition machinery (`chordal_extension`, `chordal_transform`,
  `is_chordal`, …) moved to NonconvexSemidefinite.jl; Core no longer
  depends on AMD.jl.
- Removed legacy APIs: `JuMPSymbolicFunction` and `jump_symbolic_*` (JuMP
  model conversion keeps working through the shared `JuMPEvaluator`),
  `setmin!`/`setmax!`/`setbox!`, the `sd` keyword of
  `getobjectiveconstraints` and the internal `debugging` flag.

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
