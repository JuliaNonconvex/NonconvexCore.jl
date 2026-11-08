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
    else
        _, J = DI.value_and_jacobian(f, backend, x)
        return y, Adjoint(Matrix(transpose(J)))
    end
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

Returns the value and transpose of the Jacobian of `f` at the point `x`. This calls [`value_jacobian`](@ref) and transposes the Jacobian.
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

# Extract the cotangent of `x` from a pullback seed application in a way that
# works across Zygote versions (older versions return the cotangent directly
# for single-argument functions while newer versions wrap it in a 1-tuple).
@inline function _pullback_cotangent(pb, seed)
    Δ = pb(seed)
    return Δ isa Tuple ? Δ[1] : Δ
end

# Reverse-mode differentiation of a `VectorOfFunctions` call as a whole gives
# wrong cotangents for Zygote: the pullback of `vcat` over scalar outputs
# passes the full seed to every function. Assemble the pullback from
# per-function pullbacks instead.
function ChainRulesCore.rrule(::Zygote.ZygoteRuleConfig, c::VectorOfFunctions, x)
    isempty(c.fs) && return (Float64[], _ -> (NoTangent(), NoTangent()))
    outs_pbs = map(c.fs) do f
        Zygote.pullback(f, x)
    end
    y = mapreduce(vcat, outs_pbs; init = Float64[]) do out_pb
        out = out_pb[1]
        out isa Real ? [out] : out
    end
    function vof_pullback(Δ)
        Δx = nothing
        offset = 0
        for (f, (out, pb)) in zip(c.fs, outs_pbs)
            dim = getdim(f)
            δ = dim == 1 ? Δ[offset+1] : Δ[offset+1:offset+dim]
            offset += dim
            Δxi = _pullback_cotangent(pb, δ)
            Δx = Δx === nothing ? Δxi : Δx + Δxi
        end
        return (NoTangent(), Δx)
    end
    return y, vof_pullback
end
