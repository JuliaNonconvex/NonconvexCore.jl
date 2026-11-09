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
- [`value_hessian`](@ref NonconvexCore.value_hessian): Hessian of a
  scalar-valued function.

All of them take an optional backend argument defaulting to the package-wide
backend.

## Explicit preparation

The differentiation backend of a function can be prepared explicitly (chunk
selection, tape compilation, sparsity detection and coloring) before any
optimization, and the preparation is then used by default:

```julia
prepare_gradient(f, x, backend)
prepare_jacobian(f, x, backend)
prepare_hessian(f, x, backend)
prepare_sparse_jacobian(f, x, backend)
prepare_sparse_hessian(f, x, backend)
```

`x` is a representative point at which the preparation is computed; the
prepared objects are cached per `(f, backend)` and consulted by
[`value_gradient`](@ref NonconvexCore.value_gradient),
[`value_jacobian`](@ref NonconvexCore.value_jacobian),
[`value_hessian`](@ref NonconvexCore.value_hessian),
[`sparse_jacobian`](@ref NonconvexCore.sparse_jacobian) and
[`sparse_hessian`](@ref NonconvexCore.sparse_hessian) thereafter.

## The `diffy` modifier

[`diffy`](@ref NonconvexCore.diffy) wraps a function so that it evaluates
through an eagerly prepared backend, and any ChainRules-compatible outer
differentiation uses that prepared backend:

```julia
F = diffy(f, x, backend)
F(x)
```

`F` can be used in place of `f` in objectives and constraints.

## Second-order composition

When a backend only provides first-order derivatives (such as a reverse-mode
backend), Hessians are computed by wrapping it in a
`DifferentiationInterface.SecondOrder` composition. The default composition is
forward-over-reverse: `SecondOrder(AutoForwardDiff(), backend)`, provided by
[`second_order_backend`](@ref NonconvexCore.second_order_backend). Overload it
to customize the Hessian backend for a given first-order backend.
