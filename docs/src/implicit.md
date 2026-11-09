# Implicit functions

An [`ImplicitFunction`](@ref) differentiates a function `y(x)` defined
implicitly by a condition `f(x, y) = 0`, using the implicit function theorem.

```julia
F = ImplicitFunction(forward, conditions; tol = 1e-5, matrixfree = false)
```

- `forward(x)` returns a tuple `(y_star, df_dy)` such that `f(x, y_star) =
  0`. `df_dy` is optional: pass `nothing` to compute the conditions Jacobian
  by automatic differentiation.
- `conditions(x, y)` is the condition function `f(x, y)`, which must be `0`
  at `y_star`. Variables that are closed over rather than passed through `x`
  are assumed to have no effect on the solution.
- `matrixfree = true` solves the implicit-function linear systems with a
  matrix-free Krylov method instead of a direct factorization.
- `jac_backend` selects the ADTypes backend used to compute `df/dy` when
  `forward` does not return it (default `AutoZygote()`).
- `tol` is the acceptable tolerance for `f(x, y_star)` to apply the implicit
  function theorem at `x`; `error_on_tol_violation` controls whether a
  tolerance violation errors or returns `NaN` gradients.

`F` differentiates correctly under any ChainRules-compatible backend.
