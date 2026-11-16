# ADNLPModels compatibility layer.
#
# Algorithm packages must not define their own AD code: this extension
# provides the ADNLPModels AD-backend interface (used by Percival-based
# wrappers) on top of NonconvexCore's backend-agnostic AD utilities
# (`value_gradient`, `value_jacobian`, `value_hessian`).
#
# The type names keep ADNLPModels' `Zygote*` convention so existing
# consumers keep working unchanged, but the derivatives are computed with
# NonconvexCore's backend-agnostic AD utilities. The compat layer uses
# `AutoForwardDiff` for every derivative:
# - ADNLPModels' `get_c` wrappers fill an internal buffer (`nlp.c!(c, x)`),
#   which reverse-mode backends (Zygote, ReverseDiff) reject as in-place
#   mutation, so first-order derivatives also go through forward mode;
# - forward-over-forward second derivatives avoid the nested-AD pitfalls of
#   Zygote/ReverseDiff (the previous nested-Zygote Hessian silently returned
#   wrong values);
# - forward mode differentiates the closed-form (affine/quadratic) channel
#   of JuMP-converted models directly, without relying on rrules.
module NonconvexCoreADNLPModelsExt

using NonconvexCore
using ADNLPModels: ADNLPModels
using ADNLPModels: ADBackend, ImmutableADbackend

# ---------------------------------------------------------------------------
# Gradient backends
# ---------------------------------------------------------------------------

struct ZygoteGradient <: ADBackend end
ZygoteGradient(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGradient()

function ADNLPModels.gradient(::ZygoteGradient, f, x)
    return NonconvexCore.value_gradient(f, x, AutoForwardDiff())[2]
end
function ADNLPModels.gradient!(::ZygoteGradient, g, f, x)
    g .= NonconvexCore.value_gradient(f, x, AutoForwardDiff())[2]
    return g
end

# ---------------------------------------------------------------------------
# Jacobian backends
# ---------------------------------------------------------------------------

struct ZygoteJacobian <: ImmutableADbackend
    nnzj::Int
end
function ZygoteJacobian(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...)
    return ZygoteJacobian(nvar * ncon)
end
function ADNLPModels.jacobian(::ZygoteJacobian, c, x)
    return NonconvexCore.value_jacobian(c, x, AutoForwardDiff())[2]
end

struct ZygoteJprod <: ImmutableADbackend end
ZygoteJprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJprod()
function ADNLPModels.Jprod!(::ZygoteJprod, Jv, c, x, v, ::Val)
    Jv .= NonconvexCore.value_jacobian(c, x, AutoForwardDiff())[2] * v
    return Jv
end

struct ZygoteJtprod <: ImmutableADbackend end
ZygoteJtprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJtprod()
function ADNLPModels.Jtprod!(::ZygoteJtprod, Jtv, c, x, v, ::Val)
    Jtv .= NonconvexCore.value_jacobian(c, x, AutoForwardDiff())[2]' * v
    return Jtv
end

# ---------------------------------------------------------------------------
# Second-order backends
# ---------------------------------------------------------------------------

struct ZygoteHessian <: ImmutableADbackend
    nnzh::Int
end
function ZygoteHessian(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...)
    return ZygoteHessian(div(nvar * (nvar + 1), 2))
end
function ADNLPModels.hessian(::ZygoteHessian, f, x)
    return NonconvexCore.value_hessian(f, x, AutoForwardDiff())
end

struct ZygoteHvprod <: ADBackend end
ZygoteHvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteHvprod()
# ADNLPModels folds the objective weight (and, for `:lag`, the constraint
# multipliers) into the Lagrangian closure `ℓ` that it passes here (see
# `ADNLPModels.get_lag`), so the closures must be differentiated as-is.
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:obj}, obj_weight)
    Hv .= NonconvexCore.value_hessian(ℓ, x, AutoForwardDiff()) * v
    return Hv
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:lag}, y, obj_weight)
    Hv .= NonconvexCore.value_hessian(ℓ, x, AutoForwardDiff()) * v
    return Hv
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:ci})
    Hv .= NonconvexCore.value_hessian(ℓ, x, AutoForwardDiff()) * v
    return Hv
end

struct ZygoteGHjvprod <: ImmutableADbackend end
ZygoteGHjvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGHjvprod()
function ADNLPModels.directional_second_derivative(::ZygoteGHjvprod, f, x, v, w)
    # Per-row directional second derivative wᵀ∇²f(x)v for scalar-valued f.
    return dot(w, NonconvexCore.value_hessian(f, x, AutoForwardDiff()) * v)
end

end
