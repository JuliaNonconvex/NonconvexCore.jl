# NEWS

## 2.3.1

- Docs: describe the package on the introduction page; drop the quick-start section from the README.

## 2.3.0

- Function-in-set constraints and variable-in-set blocks, the
  MathOptInterface-style modeling core: `FunctionInSet(f, set)` with the
  sets `EqualTo`, `LessThan`, `GreaterThan`, `Interval`, `Nonnegatives`,
  `Nonpositives`, `Zeros`, `Reals`, `Simplex`, `PSDCone`, `ChanceSet` and
  `SetProduct` — a `SetProduct` routes one bundle evaluation into several
  constraints of different kinds (scalar, vector and matrix-valued), so an
  expensive computation (e.g. a finite-element solve) feeds them all at
  once. `add_set_constraint!` and `add_chance_constraint!` register them
  (chance constraints take a user-supplied scenario batch and
  `epsilon = Zero()` for the deterministic case — Core defines the
  constraint, it does not sample); `add_set_variable!(m, set)` registers
  variable blocks (a `PSDCone` block is a semidefinite matrix variable, a
  `Simplex` block a probability-simplex variable) whose cone parts become
  synthetic set constraints at model conversion.
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
  unsupported set features with an informative error. The semidefinite
  algorithms (chordal decomposition, the structured domain decomposition,
  barrier and cutting-plane solvers) live in NonconvexSemidefinite.jl,
  while the cone type `PSDCone` itself is defined here in Core.
- Domain transforms: `nullspace_transform` and `unconstrained_transform`
  return `(model, x0, mapper)` and differentiate through the mapping with
  analytic ChainRules rrules (TransformVariables + Zygote), including a
  Stan-style simplex.
- Epigraph reformulation: `add_epigraph_variable!` and `epigraph!`.
- First-class linear constraints: `LinearFunction` and
  `add_linear_constraint!`.
- The chordal-decomposition machinery (`chordal_extension`, `chordal_transform`,
  `is_chordal`, …) moved to NonconvexSemidefinite.jl; Core no longer
  depends on AMD.jl.
- Removed legacy APIs: `JuMPSymbolicFunction` and `jump_symbolic_*` (JuMP
  model conversion keeps working through the shared `JuMPEvaluator`),
  `setmin!`/`setmax!`/`setbox!`, the `sd` keyword of
  `getobjectiveconstraints` and the internal `debugging` flag.


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
