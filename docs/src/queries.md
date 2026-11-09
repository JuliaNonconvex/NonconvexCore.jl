# Model queries

There are a number of things you can query about a model after constructing
it. These can be useful to check that the model was defined correctly, or in
post-processing after optimization.

- [`getmin`](@ref) returns the lower bounds of the variables; with an index
  argument, the lower bound of one variable.
- [`getmax`](@ref) returns the upper bounds of the variables; with an index
  argument, the upper bound of one variable.
- [`getinit`](@ref NonconvexCore.getinit) returns the initial solution.
- [`getobjective`](@ref) returns the objective function.
- [`getineqconstraints`](@ref NonconvexCore.getineqconstraints) returns the
  collection of inequality constraint functions.
- [`geteqconstraints`](@ref NonconvexCore.geteqconstraints) returns the
  collection of equality constraint functions.
- [`getobjectiveconstraints`](@ref NonconvexCore.getobjectiveconstraints)
  returns the objective and constraint functions together, optionally
  including the semidefinite constraints with `sd = true`.
- [`getnvars`](@ref NonconvexCore.getnvars) returns the number of variables.
- [`getnconstraints`](@ref NonconvexCore.getnconstraints) returns the number
  of constraints.

The [`setmin!`](@ref), [`setmax!`](@ref) and [`setinteger!`](@ref) functions
update the bounds and integrality constraints after construction, which is
useful for iterative procedures that resolve the same model with updated
bounds.
