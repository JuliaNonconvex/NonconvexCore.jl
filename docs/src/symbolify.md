# Symbolic differentiation

For functions with a tractable symbolic form, [`symbolify`](@ref) computes
the gradient, Jacobian and/or Hessian symbolically with
[Symbolics](https://github.com/JuliaSymbolics/Symbolics.jl). Symbolic
simplifications and cancellations can sometimes be cheaper than algorithmic
differentiation, and symbolic differentiation can exploit derivative
sparsity.

> Symbolics is an optional dependency of NonconvexCore: `symbolify` is
> provided by the `NonconvexCoreSymbolicsExt` package extension. To use it,
> load Symbolics first (e.g. `using Symbolics`) so the extension activates.

## Symbolifying a function

```julia
F = symbolify(f, x...; hessian = false, sparse = false, simplify = false)
F(x...)
```

where `x` are some sample input arguments to `f`. `F(x...)` can be used in
place of `f(x...)` in objectives and/or constraints. Whenever a
ChainRules-compatible AD package such as Zygote differentiates `F`, the
Symbolics-derived gradient/Jacobian is used.

- `sparse = true` tells Symbolics to return a sparse gradient/Jacobian.
- `simplify = true` asks Symbolics to simplify the derivative expressions.
- `hessian = true` additionally computes the Hessian symbolically (for
  vector-valued functions, the Hessian of the flattened outputs).

## Symbolifying a model

Instead of symbolifying one function at a time, an entire model (objective
and all constraints) can be symbolified:

```julia
sym_model = symbolify(model; hessian = true, simplify = true, sparse = true)
```

By default, the objective and all the constraint functions are symbolified.
Any of the following keyword arguments can be set to `false` (default
`true`) to leave that component untouched:

- `objective = false`
- `ineq_constraints = false`
- `eq_constraints = false`
- `sd_constraints = false`

## `sparse = true` vs the sparse derivative pipeline

`symbolify(; sparse = true)` and NonconvexCore's
[sparse derivative pipeline](derivatives/sparse.md)
([`sparse_jacobian`](@ref NonconvexCore.sparse_jacobian) and
[`sparse_hessian`](@ref NonconvexCore.sparse_hessian)) both compute sparse
derivatives, but by different means:

- `symbolify(; sparse = true)` is **symbolic**: Symbolics constructs the
  exact mathematical expression of the gradient/Jacobian/Hessian of the
  function, simplifies it when `simplify = true`, and returns the derivative
  in sparse form. Terms that cancel mathematically are removed by the
  simplification, and the derivatives are then evaluated by the generated
  functions.
- The sparse derivative pipeline is **algorithmic**: it traces the function
  with SparseConnectivityTracer to detect the *structural* sparsity pattern,
  colors the pattern, and computes the entries with an automatic
  differentiation backend. The pattern is structural: an entry that is
  structurally present but mathematically cancels (e.g. in
  `sin(x)^2 + cos(x)^2 - 1`) is still computed. No symbolic simplification
  happens, but the pipeline works for any function the tracer and the AD
  backend can handle, not only functions Symbolics can understand.

In short: `symbolify` pays a one-time symbolic cost to get exact, simplified,
sparse derivative functions; the sparse pipeline pays a one-time tracing and
coloring cost to get structural sparsity for arbitrary code. Both
preparations are cached and can be done explicitly up front (see
[Explicit preparation](derivatives/backends.md#Explicit-preparation)).
