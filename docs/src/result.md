# Optimization result

Each solver is free to return a different result type from `optimize`.
However, all the result types have 2 fields:

- `result.minimum`: the minimum objective value reached in the optimization.
- `result.minimizer`: the optimal decision variables reached during
  optimization.

Some result types store additional information returned by the solver, e.g.
the convergence status. Please explore the fields of the `result` output
from `optimize` and check the documentation of the individual solver
packages in the [Nonconvex documentation](https://julianonconvex.github.io/Nonconvex.jl/).
