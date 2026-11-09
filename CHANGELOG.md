# Changelog

## 2.1.0 (unreleased)

- Type-stability, Aqua and JET checks for the main functionality, with
  fixes for the issues they surfaced.
- The codebase is formatted with JuliaFormatter (BlueStyle) and a
  format-check CI workflow enforces it.

## 2.0.0

### Breaking changes

- **Solver capability system is strict**: `ConstraintCapabilities` now
  declares support separately for **linear/nonlinear** equality, inequality
  and semidefinite constraints, plus **continuous / binary / integer**
  variables, box bounds and infinite bounds. Constraint linearity is exact
  for JuMP-converted models and Hessian-pattern-based otherwise
  (conservatively nonlinear when untraceable). `checksubproblem!` errors on
  undeclared or unsupported features. Solver packages must update their
  `solver_capabilities` declarations.
- `value_jacobian` returns Jacobians without `Adjoint`/`transpose` wrapping:
  dense for dense derivatives, **sparse for sparse ones**; scalar gradients
  are reshaped to `1 x inputs` matrices.
- The default AD backend is now `AutoZygote()` (changeable with
  `set_ad_backend!`); Zygote's array-pullback double-accumulation through
  flattened containers is worked around by row-by-row Jacobian assembly.
- **NonconvexUtils is absorbed**: `CustomGradFunction`,
  `CustomHessianFunction`, `TraceFunction`, `LazyJacobian` and
  `@ForwardDiff_frule` now live here; `symbolify` is behind the
  `NonconvexCoreSymbolicsExt` extension (load Symbolics). `diffiy`,
  `sparsify` and `ImplicitFunction` were dropped as redundant/unused.
- JuMP is an **optional dependency** (`NonconvexCoreJuMPExt`),
  ConvergencePlots an optional dependency
  (`NonconvexCoreConvergencePlotsExt`; the `Requires` dependency is gone),
  and ReverseDiff a weakdep (load it to select `AutoReverseDiff()`).
- `get_jump_problem`'s evaluator implements Jac-vec and Hess-vec products
  (`:JacVec`, `:HessVec`).

### Highlights

- **JuMP conversion** `DictModel(::JuMP.Model)`: affine/quadratic models use
  the exact matrix path; nonlinear models are evaluated by **one shared
  `MOI.Nonlinear` evaluator** (`JuMPEvaluator`) — values, sparse Jacobian and
  Lagrangian Hessian each from a single evaluator pass (6x–760x faster than
  the native closure path on the Rosetta AC-OPF cases 14/118/300).
  `JuMPSymbolicFunction` evaluates single MOI expressions.
- **Explicit preparation**: `prepare_gradient` / `prepare_jacobian` /
  `prepare_hessian` / `prepare_sparse_*` / `prepare_model` (with
  `gradient`/`hessian`/`jacobian` options) prepare backends up front, cached
  per `(f, backend)` and used by default; `diffy` wraps a function in an
  eagerly prepared backend.
- **Sparse derivatives**: `sparse_jacobian` / `sparse_hessian` via
  SparseConnectivityTracer detection + SparseMatrixColorings coloring +
  colored DifferentiationInterface evaluation, with `second_order_backend`
  composition; failures raise loudly.

### Documentation & tests

- A **DocumenterFragments** docs fragment covering models, the JuMP
  conversion, AD backends and preparation, the sparse pipeline, function
  modifiers, symbolic differentiation, capabilities, criteria, queries and
  results, with `checkdocs = :all` and extension-activation notes.
- The test suite is organized into granular **TestItems** for isolated and
  parallel debugging; coverage raised from ~41% to near-total.

### Issues closed

- #33 (unbounded-variable diagnostics), #37 (Jac-vec products), #38
  (coverage).
