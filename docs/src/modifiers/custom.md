# Custom derivatives

## Gradients and Jacobians

To use a user-defined gradient/Jacobian function `g(x)` for a function
`f(x)`, use the [`CustomGradFunction`](@ref) modifier:

```julia
F = CustomGradFunction(f, g)
F(x)
```

`F` can be used in place of `f` as an objective function, as a constraint
function or as part of any such function. When `f` is scalar-valued, `g` is
expected to return a gradient vector. When `f` is vector-valued, `g` is
expected to return a Jacobian matrix. Whenever a ChainRules-compatible AD
package such as Zygote differentiates `F`, the custom gradient/Jacobian `g`
is used.

## Hessians and Hessian-vector products

For second-order optimization algorithms, a user-defined Hessian function
`h(x)` can be used for any scalar-valued function `f(x)` with gradient
`g(x)`. To use a user-defined Hessian function `h(x)`, use the
[`CustomHessianFunction`](@ref) modifier:

```julia
F = CustomHessianFunction(f, g, h)
F(x)
```

`f` is expected to return a scalar, `g` a gradient vector and `h` a symmetric
Hessian matrix. Whenever a ChainRules-compatible AD package differentiates
`F`, the custom gradient and Hessian functions are used.

Instead of a Hessian function `h`, a Hessian-vector product operator
`hvp(x, v)` can be used, which multiplies the Hessian of `f` at `x` by the
vector `v`. To use a Hessian-vector product operator instead of the full
Hessian, pass the `hvp` function as the third argument and set the `hvp`
keyword argument to `true`:

```julia
F = CustomHessianFunction(f, g, hvp; hvp = true)
F(x)
```
