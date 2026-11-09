# Explicit preparation of the DifferentiationInterface backends. Prepared
# objects are cached by `(f, backend)` and then used by default by the
# value-and-derivative functions (value_gradient, value_jacobian,
# value_hessian) and the sparse derivative functions (sparse_jacobian,
# sparse_hessian).

const _PREP_CACHES = (
    gradient = Dict{Any,Any}(),
    jacobian = Dict{Any,Any}(),
    hessian = Dict{Any,Any}(),
)

"""
    prepare_gradient(f, x, [backend])
    prepare_jacobian(f, x, [backend])
    prepare_hessian(f, x, [backend])

Explicitly prepares the differentiation backend `backend` (an ADTypes
backend, defaulting to the package-wide backend) for `f` at the
representative point `x`. The preparation (chunk selection, tape
compilation, sparsity detection and coloring, ...) is cached by
`(f, backend)` and used by default by [`value_gradient`](@ref NonconvexCore.value_gradient),
[`value_jacobian`](@ref NonconvexCore.value_jacobian) and
[`value_hessian`](@ref NonconvexCore.value_hessian) afterwards.
"""
function prepare_gradient(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    prep = get(_PREP_CACHES.gradient, key, nothing)
    prep === nothing || return prep
    prep = DI.prepare_gradient(f, backend, x)
    _PREP_CACHES.gradient[key] = prep
    return prep
end

function prepare_jacobian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    prep = get(_PREP_CACHES.jacobian, key, nothing)
    prep === nothing || return prep
    prep = DI.prepare_jacobian(f, backend, x)
    _PREP_CACHES.jacobian[key] = prep
    return prep
end

function prepare_hessian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    prep = get(_PREP_CACHES.hessian, key, nothing)
    prep === nothing || return prep
    prep = DI.prepare_hessian(f, backend, x)
    _PREP_CACHES.hessian[key] = prep
    return prep
end

"""
    prepare_sparse_jacobian(f, x, [backend])
    prepare_sparse_hessian(f, x, [backend])

Explicitly prepares the sparse derivative pipeline (sparsity detection with
SparseConnectivityTracer, coloring with SparseMatrixColorings and the colored
DifferentiationInterface evaluation) for `f` at the representative point `x`.
The preparation is cached by `(f, backend)` and used by default by
[`sparse_jacobian`](@ref NonconvexCore.sparse_jacobian) and
[`sparse_hessian`](@ref NonconvexCore.sparse_hessian).
"""
function prepare_sparse_jacobian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    entry = get(_SPARSE_JACOBIAN_PREP_CACHE, key, nothing)
    entry === nothing || return entry
    y = f(x)
    f_ = y isa Number ? (z -> [f(z)]) : f
    sb = _sparse_backend(backend)
    entry = (f_, sb, DI.prepare_jacobian(f_, sb, x))
    _SPARSE_JACOBIAN_PREP_CACHE[key] = entry
    return entry
end

function prepare_sparse_hessian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    entry = get(_SPARSE_HESSIAN_PREP_CACHE, key, nothing)
    entry === nothing || return entry
    sb = _sparse_backend(second_order_backend(backend))
    entry = (sb, DI.prepare_hessian(f, sb, x))
    _SPARSE_HESSIAN_PREP_CACHE[key] = entry
    return entry
end

"""
    diffy(f, x, [backend])

Returns a callable that evaluates `f` and whose derivatives are computed by
the differentiation backend `backend` (an ADTypes backend, defaulting to the
package-wide backend), prepared eagerly at the representative point `x`. The
returned function can be used in place of `f` in objectives and constraints;
any ChainRules-compatible outer differentiation uses the prepared backend.
"""
struct PreparedFunction{F,B,P} <: Function
    f::F
    backend::B
    prep::P
end
(f::PreparedFunction)(x) = f.f(x)

function diffy(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    y = f(x)
    if y isa Number
        prep = prepare_gradient(f, x, backend)
    else
        prep = prepare_jacobian(f, x, backend)
    end
    return PreparedFunction(f, backend, prep)
end

function ChainRulesCore.rrule(g::PreparedFunction, x::AbstractVector)
    y = g.f(x)
    if y isa Number
        return y, Δ -> (NoTangent(), value_gradient(g.f, x, g.backend)[2] .* _unthunk(Δ))
    end
    J = value_jacobian(g.f, x, g.backend)[2]
    return y, Δ -> (NoTangent(), Matrix(J)' * _unthunk(Δ))
end

"""
    prepare_model(model, [x], [backend]; gradient = true, hessian = true, jacobian = true)

Explicitly prepares the differentiation backend for the selected pieces of
`model` at the initial point, and returns the prepared vectorized model:

- `gradient`: the objective gradient,
- `hessian`: the objective Hessian (not always required, e.g. for
  first-order solvers),
- `jacobian`: the Jacobians of all the constraints.

The preparations are cached per function and backend and are used by default
by the value-and-derivative and sparse derivative functions on the returned
model.
"""
function prepare_model(
    model::AbstractModel,
    x = getinit(model),
    backend::AbstractADType = DEFAULT_AD_BACKEND[];
    gradient = true,
    hessian = true,
    jacobian = true,
)
    vm, xv, _ = tovecmodel(model, x)
    obj = getobjective(vm)
    gradient && prepare_gradient(obj, xv, backend)
    hessian && prepare_hessian(obj, xv, backend)
    if jacobian
        for c in getineqconstraints(vm).fs
            prepare_jacobian(c, xv, backend)
        end
        for c in geteqconstraints(vm).fs
            prepare_jacobian(c, xv, backend)
        end
    end
    return vm
end

"""
    diffy(f, x, [backend])

Returns a callable that evaluates `f` and differentiates through an eagerly
prepared backend. See [`prepare_gradient`](@ref NonconvexCore.prepare_gradient)
and [`prepare_jacobian`](@ref NonconvexCore.prepare_jacobian).
"""
diffy

"""
    prepare_gradient(f, x, [backend])

Explicitly prepares the gradient backend of `f`; cached and used by default.
"""
prepare_gradient

"""
    prepare_jacobian(f, x, [backend])

Explicitly prepares the Jacobian backend of `f`; cached and used by default.
"""
prepare_jacobian

"""
    prepare_hessian(f, x, [backend])

Explicitly prepares the Hessian backend of `f`; cached and used by default.
"""
prepare_hessian

"""
    prepare_sparse_jacobian(f, x, [backend])

Explicitly prepares the sparse Jacobian pipeline of `f`; cached and used by
default by `sparse_jacobian`.
"""
prepare_sparse_jacobian

"""
    prepare_sparse_hessian(f, x, [backend])

Explicitly prepares the sparse Hessian pipeline of `f`; cached and used by
default by `sparse_hessian`.
"""
prepare_sparse_hessian

