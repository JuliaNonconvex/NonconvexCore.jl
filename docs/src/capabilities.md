# Solver capabilities

Solver packages declare which models they support through
[`ConstraintCapabilities`](@ref):

```julia
ConstraintCapabilities(; eq, ineq, sd, integer, infinite_bounds)
```

The fields (each `true`, `false` or `missing` for "not declared") are:

- `eq`: equality constraints,
- `ineq`: inequality constraints,
- `sd`: semidefinite constraints,
- `integer`: integer variables,
- `infinite_bounds`: variables with infinite (unbounded) bounds.

[`solver_capabilities`](@ref NonconvexCore.solver_capabilities) queries a
solver's declared capabilities, [`merge_capabilities`](@ref) combines
capability declarations (for meta-algorithms that forward their
sub-solvers' capabilities), and [`checksubproblem!`](@ref) validates a model
against the capabilities of the selected solver before optimization.
