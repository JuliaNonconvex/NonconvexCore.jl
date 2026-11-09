# Solver capabilities

Solver packages declare which models they support through
[`ConstraintCapabilities`](@ref):

```julia
ConstraintCapabilities(; linear_eq, nonlinear_eq, linear_ineq, nonlinear_ineq,
                       linear_sd, nonlinear_sd, box, continuous, binary,
                       integer, infinite_bounds)
```

The fields (each `true`, `false` or `missing` for "not declared") are:

- `linear_eq` / `nonlinear_eq`: linear / nonlinear equality constraints,
- `linear_ineq` / `nonlinear_ineq`: linear / nonlinear inequality constraints,
- `linear_sd` / `nonlinear_sd`: linear / nonlinear semidefinite constraints,
- `box`: finite variable bounds,
- `continuous`, `binary`, `integer`: continuous, binary (0/1) and integer
  (non-binary) variables,
- `infinite_bounds`: variables with infinite (unbounded) bounds.

Constraint linearity is detected exactly for JuMP-converted models (from the
expression graphs) and via the structural Hessian for other models: a
function is structurally linear exactly when its Hessian pattern is empty.
Functions that cannot be traced are conservatively classified as nonlinear.

[`solver_capabilities`](@ref NonconvexCore.solver_capabilities) queries a
solver's declared capabilities, [`merge_capabilities`](@ref) combines
capability declarations (for meta-algorithms that forward their
sub-solvers' capabilities), and [`checksubproblem!`](@ref) validates a model
against the capabilities of the selected solver before optimization.
