# Convergence criteria

NonconvexCore provides convergence criteria that solver packages can use to
check the quality of a solution against a model.

A [`Workspace`](@ref) collects the workspace required to evaluate a
criterion, and the criteria themselves are:

- [`GenericCriteria`](@ref): convergence based on the change in the solution
  and the objective, and the constraint violation.
- [`KKTCriteria`](@ref) and [`ScaledKKTCriteria`](@ref): convergence based on
  the KKT residuals and the constraint violation.
- [`IpoptCriteria`](@ref): convergence based on Ipopt's optimality error and
  the constraint violation.

The thresholds of a criterion are configured with [`Tolerance`](@ref).
