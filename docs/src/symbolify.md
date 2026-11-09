# Symbolic differentiation

For functions with a tractable symbolic form, [`symbolify`](@ref) computes
the gradient, Jacobian and/or Hessian symbolically with
[Symbolics](https://github.com/JuliaSymbolics/Symbolics.jl). Symbolic
simplifications and cancellations can sometimes be cheaper than algorithmic
differentiation, and symbolic differentiation can exploit derivative
sparsity.

> Symbolics is an optional dependency of NonconvexCore: `symbolify` is
> provided by the `NonconvexCoreSymbolicsExt` package extension, which loads
> automatically when Symbolics is present in the environment.

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
