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
function value_jacobian(f::Function, x::AbstractVector, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    y = f(x)
    if y isa Number
        _, g = DI.value_and_gradient(f, backend, x)
        return y, g'
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
    _, J = DI.value_and_jacobian(f, backend, x)
    return y, Adjoint(Matrix(transpose(J)))
end

function _value_jacobian_zygote_rows(f, x, y, backend)
    m = length(y)
    J = Matrix{Float64}(undef, m, length(x))
    for k = 1:m
        _, g = DI.value_and_gradient(z -> f(z)[k], backend, x)
        J[k, :] .= g
    end
    return y, Adjoint(transpose(J))
end
function value_jacobian(c::VectorOfFunctions, x::AbstractVector, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    isempty(c.fs) && return (c(x), adjoint(zeros(length(x), 0)))
    # Differentiate each function in the collection separately: differentiating
    # the concatenated call `c(x)` as a whole is not reliable for every backend
    # (e.g. Zygote's pullback of `vcat` over scalar outputs).
    vals_jacs = map(c.fs) do f
        value_jacobian(f, x, backend)
    end
    vals = mapreduce(vcat, vals_jacs; init = zeros(0)) do val_jac
        val_jac[1]
    end
    jact = mapreduce(hcat, vals_jacs; init = zeros(length(x), 0)) do val_jac
        val_jac[2]'
    end
    return vals, jact'
end

"""
    value_jacobian_transpose(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the value and transpose of the Jacobian of `f` at the point `x`. This calls [`value_jacobian`](@ref NonconvexCore.value_jacobian) and transposes the Jacobian.
"""
function value_jacobian_transpose(f::Function, x::AbstractVector, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    val, jac = value_jacobian(f, x, backend)
    return val, jac'
end

"""
    value_gradient(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the value and gradient of the scalar-valued function `f` at the point `x`, computed with the automatic differentiation backend `backend` (defaults to the package-wide backend returned by [`get_ad_backend`](@ref)).
"""
function value_gradient(f::Function, x::AbstractVector, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    return DI.value_and_gradient(f, backend, x)
end

"""
    value_hessian(f::Function, x::AbstractVector, [backend::ADTypes.AbstractADType])

Returns the Hessian of the scalar-valued function `f` at the point `x`, computed with the automatic differentiation backend `backend` (defaults to the package-wide backend returned by [`get_ad_backend`](@ref)).
"""
value_hessian(f::Function, x::AbstractVector, backend::AbstractADType = DEFAULT_AD_BACKEND[]) =
    DI.hessian(f, backend, x)



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
    value_jacobian_transpose(f, x, [backend])

Returns the value and the transpose of the Jacobian of `f` at `x`.
"""
value_jacobian_transpose

"""
    value_hessian(f, x, [backend])

Returns the Hessian of the scalar-valued function `f` at `x`.
"""
value_hessian
