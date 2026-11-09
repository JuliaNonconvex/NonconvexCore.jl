# AD backends

NonconvexCore computes derivatives through the
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
(ADTypes) backend system.

## The default backend

The package-wide default backend is `AutoZygote()`: Zygote's ChainRules-based
rules support arbitrary argument types (dictionaries, structs, arrays)
through the model-flattening machinery. It can be changed with
[`set_ad_backend!`](@ref) and queried with [`get_ad_backend`](@ref):

```julia
set_ad_backend!(ADTypes.AutoReverseDiff())
get_ad_backend()
```

The package providing the backend must be loaded by the caller (for example,
load ReverseDiff before selecting `AutoReverseDiff()`).

## Value and derivative functions

The generic derivative entry points are:

- [`value_gradient`](@ref NonconvexCore.value_gradient): value and gradient of
  a scalar-valued function.
- [`value_jacobian`](@ref NonconvexCore.value_jacobian): value and Jacobian of
  a (possibly vector-valued) function or a `VectorOfFunctions` collection.
- [`value_jacobian_transpose`](@ref NonconvexCore.value_jacobian_transpose):
  value and the transpose of the Jacobian.
- [`value_hessian`](@ref NonconvexCore.value_hessian): Hessian of a
  scalar-valued function.

All of them take an optional backend argument defaulting to the package-wide
backend.

## Second-order composition

When a backend only provides first-order derivatives (such as a reverse-mode
backend), Hessians are computed by wrapping it in a
`DifferentiationInterface.SecondOrder` composition. The default composition is
forward-over-reverse: `SecondOrder(AutoForwardDiff(), backend)`, provided by
[`second_order_backend`](@ref NonconvexCore.second_order_backend). Overload it
to customize the Hessian backend for a given first-order backend.
