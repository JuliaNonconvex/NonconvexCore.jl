# Lazy Jacobians

A [`LazyJacobian`](@ref) represents a Jacobian by its matrix-free operators
instead of a dense or sparse matrix, which is useful when the Jacobian is
large but its Jacobian-vector and vector-Jacobian products are cheap.

```julia
LazyJacobian(; jvp = (v) -> ..., jtvp = (w) -> ..., symmetric = false)
```

- `jvp(v)` computes the Jacobian-vector product `J * v`.
- `jtvp(w)` computes the vector-Jacobian product `J' * w`.
- when `symmetric = true`, the matrix is symmetric and only one of the two
  operators needs to be supplied.

`LazyJacobian` overloads `adjoint`, `transpose` and `*`, so it can be used
wherever a matrix is expected, including inside the derivative functions of
NonconvexCore and any ChainRules-based outer differentiation.
