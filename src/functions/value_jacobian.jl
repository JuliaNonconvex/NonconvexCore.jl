# Simultaneous function and Jacobian evaluation

"""
    value_jacobian(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the value and Jacobian of the function `f` at the point `x`. The
automatic differentiation backend is `backend`, an instance of
`ADTypes.AbstractADType` such as `AutoZygote()`, `AutoForwardDiff()`,
`AutoEnzyme()` or `AutoMooncake()`; it defaults to the package-wide backend
returned by [`get_ad_backend`](@ref). Custom adjoint rules for any backend can
be defined with
[`ChainRulesCore.jl`](https://github.com/JuliaDiff/ChainRulesCore.jl). The
Jacobian is returned as an instance of `Adjoint` for cache efficiency in the
optimizer.
"""
function value_jacobian(
    f::Function, x::AbstractVector, backend::AbstractADType=DEFAULT_AD_BACKEND[]
)
    y = f(x)
    if y isa Number
        _, g = DI.value_and_gradient(f, backend, x)
        return y, reshape(g, 1, :)
    end
    if backend isa AutoZygote
        # Zygote's pullback of array-valued outputs can double-accumulate
        # entries when the function reconstructs flattened containers (e.g.
        # dictionaries); each scalar row differentiates exactly, so assemble
        # the Jacobian row by row.
        g = f
        while g isa AbstractFunction && !(g isa FlattenedFunction)
            g = getfunction(g)
        end
        if g isa FlattenedFunction
            return _value_jacobian_zygote_rows(f, x, y, backend)
        end
    end
    prep = get(_PREP_CACHES.jacobian, (f, backend), nothing)
    if prep === nothing
        _, J = DI.value_and_jacobian(f, backend, x)
    else
        _, J = DI.value_and_jacobian(f, prep, backend, x)
    end
    return y, J
end

function _value_jacobian_zygote_rows(f, x, y, backend)
    m = length(y)
    J = Matrix{Float64}(undef, m, length(x))
    for k in 1:m
        _, g = DI.value_and_gradient(z -> f(z)[k], backend, x)
        J[k, :] .= g
    end
    return y, J
end
function value_jacobian(
    c::VectorOfFunctions, x::AbstractVector, backend::AbstractADType=DEFAULT_AD_BACKEND[]
)
    isempty(c.fs) && return (c(x), adjoint(zeros(length(x), 0)))
    # Differentiate each function in the collection separately: differentiating
    # the concatenated call `c(x)` as a whole is not reliable for every backend
    # (e.g. Zygote's pullback of `vcat` over scalar outputs).
    vals_jacs = map(c.fs) do f
        return value_jacobian(f, x, backend)
    end
    vals = mapreduce(vcat, vals_jacs; init=zeros(0)) do val_jac
        return val_jac[1]
    end
    J = mapreduce(vcat, vals_jacs; init=zeros(0, length(x))) do val_jac
        return reshape(val_jac[2], :, length(x))
    end
    return vals, J
end

"""
    value_gradient(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the value and gradient of the scalar-valued function `f` at the point `x`, computed with the automatic differentiation backend `backend` (defaults to the package-wide backend returned by [`get_ad_backend`](@ref)).
"""
function value_gradient(
    f::Function, x::AbstractVector, backend::AbstractADType=DEFAULT_AD_BACKEND[]
)
    prep = get(_PREP_CACHES.gradient, (f, backend), nothing)
    prep === nothing || return DI.value_and_gradient(f, prep, backend, x)
    return DI.value_and_gradient(f, backend, x)
end

"""
    value_hessian(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the Hessian of the scalar-valued function `f` at the point `x`, computed with the automatic differentiation backend `backend` (defaults to the package-wide backend returned by [`get_ad_backend`](@ref)).
"""
function value_hessian(
    f::Function, x::AbstractVector, backend::AbstractADType=DEFAULT_AD_BACKEND[]
)
    prep = get(_PREP_CACHES.hessian, (f, backend), nothing)
    prep === nothing || return DI.hessian(f, prep, backend, x)
    return DI.hessian(f, backend, x)
end

"""
    value_gradient(f, x, [backend])

Returns the value and gradient of the scalar-valued function `f` at `x`.
"""
value_gradient

"""
    value_jacobian(f, x, [backend])

Returns the value and Jacobian of the function (or `VectorOfFunctions`
collection) `f` at `x`.
"""
value_jacobian

"""
    value_hessian(f, x, [backend])

Returns the Hessian of the scalar-valued function `f` at `x`.
"""
value_hessian
