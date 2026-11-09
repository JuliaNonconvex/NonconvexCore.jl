# Sparse Jacobians and Hessians

For functions with structurally sparse derivatives, NonconvexCore provides a
sparse differentiation pipeline:

1. the sparsity pattern of the Jacobian (or Hessian) is detected with
   [SparseConnectivityTracer](https://github.com/adrhill/SparseConnectivityTracer.jl);
2. the pattern is colored with
   [SparseMatrixColorings](https://github.com/JuliaSmoothOptimizers/SparseMatrixColorings.jl);
3. the entries are computed by a colored DifferentiationInterface evaluation
   with the selected AD backend.

The entry points are [`sparse_jacobian`](@ref NonconvexCore.sparse_jacobian)
and [`sparse_hessian`](@ref NonconvexCore.sparse_hessian):

```julia
J = NonconvexCore.sparse_jacobian(f, x)
H = NonconvexCore.sparse_hessian(f, x)
```

Both accept an optional backend argument defaulting to the package-wide
backend, and their preparations (sparsity detection, coloring and, for some
backends, compiled tapes) are cached per function and backend, so the cost is
paid once per optimization run.

Sparse Hessians are prepared with the backend from
[`second_order_backend`](@ref NonconvexCore.second_order_backend).

## Fallbacks

The sparse pipeline degrades gracefully:

- when sparsity detection fails for a function (for example, functions whose
  internals branch on values cannot be traced), the Jacobian values are
  assembled from per-function derivatives at the detected pattern positions,
  or computed densely as a last resort;
- when a Hessian preparation fails, the Hessian falls back to central finite
  differences of the sparse gradient.

## Notes

The sparse pipeline is opt-in and is not always the fastest option: the
fastest derivative evaluation for a given problem depends on the function
structure and the solver. In particular, models converted from JuMP evaluate
their sparse Jacobians and Hessians directly through the
`MOI.Nonlinear` evaluator (see [JuMP models](../jump.md)), without this
pipeline. The pipeline remains available for models built with native
Nonconvex functions and will continue to be improved.
