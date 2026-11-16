# ADNLPModels compatibility layer.
#
# Algorithm packages must not define their own AD code: this extension
# provides the ADNLPModels AD-backend interface (used by Percival-based
# wrappers) on top of NonconvexCore's AD ecosystem. The type names keep
# ADNLPModels' `Zygote*` convention so existing consumers keep working
# unchanged.
#
# Derivative strategy (mirrors the historically proven ecosystem behavior,
# with the known nested-Zygote Hessian bug fixed):
# - gradients and Jacobians use Zygote's direct entry points, which consult
#   ChainRules and pick up the analytic `JuMPModel*` rrules of JuMP-converted
#   models. DifferentiationInterface's Zygote path is not used here: its
#   pullback-based Jacobian rejects the internal buffer fills of ADNLPModels'
#   out-of-place constraint wrappers (`get_c`), while `Zygote.jacobian` and
#   `Zygote.gradient` handle them (as the ecosystem's historical backends
#   did). Forward mode cannot differentiate converted models at all: the MOI
#   reverse-mode evaluator stores `Float64` internally and rejects
#   `ForwardDiff.Dual` inputs;
# - the Hessian differentiates the Zygote gradient with ForwardDiff
#   (forward-over-reverse): correct for any differentiable function, fixing
#   the previous nested-Zygote Hessian that silently returned wrong values;
# - Hessian-vector products and directional second derivatives use
#   forward-over-forward mode (raw ForwardDiff), the mutation-tolerant
#   channel; these stay outside the DifferentiationInterface pipeline
#   because nested-prep tags conflict with the wrapper closures.
module NonconvexCoreADNLPModelsExt

using NonconvexCore
using ADNLPModels: ADNLPModels
using ADNLPModels: ADBackend, ImmutableADbackend
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using ADTypes: AutoForwardDiff

# ---------------------------------------------------------------------------
# Gradient backends
# ---------------------------------------------------------------------------

struct ZygoteGradient <: ADBackend end
ZygoteGradient(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGradient()

function ADNLPModels.gradient(::ZygoteGradient, f, x)
    g = Zygote.gradient(f, x)[1]
    return g === nothing ? zero(x) : g
end
function ADNLPModels.gradient!(::ZygoteGradient, g, f, x)
    _g = Zygote.gradient(f, x)[1]
    g .= _g === nothing ? 0 : _g
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
ADNLPModels.jacobian(::ZygoteJacobian, c, x) = Zygote.jacobian(c, x)[1]

struct ZygoteJprod <: ImmutableADbackend end
ZygoteJprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJprod()
function ADNLPModels.Jprod!(::ZygoteJprod, Jv, c, x, v, ::Val)
    # Forward-mode directional Jacobian: c is vector-valued, so use a
    # one-variable jacobian (derivative requires a scalar output).
    Jv .= vec(ForwardDiff.jacobian(t -> c(x .+ t[1] .* v), [0.0]))
    return Jv
end

struct ZygoteJtprod <: ImmutableADbackend end
ZygoteJtprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJtprod()
function ADNLPModels.Jtprod!(::ZygoteJtprod, Jtv, c, x, v, ::Val)
    g = Zygote.gradient(x -> dot(c(x), v), x)[1]
    g === nothing ? (Jtv .= zero(x)) : (Jtv .= g)
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
    # Forward-over-reverse: differentiate the ChainRules-first Zygote
    # gradient with ForwardDiff. Nested Zygote (`Zygote.jacobian` of
    # `Zygote.gradient`) is not reliable — Zygote cannot differentiate
    # through its own gradient — and used to return wrong Hessians silently.
    return Matrix(ForwardDiff.jacobian(x -> begin
        g = Zygote.gradient(f, x)[1]
        return g === nothing ? zero(x) : g
    end, x))
end

struct ZygoteHvprod <: ADBackend end
ZygoteHvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteHvprod()
# ADNLPModels folds the objective weight (and, for `:lag`, the constraint
# multipliers) into the Lagrangian closure `ℓ` that it passes here (see
# `ADNLPModels.get_lag`), so the closures must be differentiated as-is.
# Forward-over-forward tolerates the internal buffers of the constraint
# wrappers and never differentiates AD through AD.
function _Hv!(Hv, ℓ, x, v)
    Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x .+ t .* v), 0)
    return Hv
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:obj}, obj_weight)
    return _Hv!(Hv, ℓ, x, v)
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:lag}, y, obj_weight)
    return _Hv!(Hv, ℓ, x, v)
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:ci})
    return _Hv!(Hv, ℓ, x, v)
end

struct ZygoteGHjvprod <: ImmutableADbackend end
ZygoteGHjvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGHjvprod()
function ADNLPModels.directional_second_derivative(::ZygoteGHjvprod, f, x, v, w)
    # Per-row directional second derivative gHv[i] = w'∇²fᵢ(x)v, mirroring
    # ADNLPModels' ForwardDiff reference backend: double forward mode.
    return ForwardDiff.derivative(
        t -> ForwardDiff.derivative(s -> f(x .+ s .* w .+ t .* v), 0.0), 0.0
    )
end

# The default `get_c` fills a plain array, which Zygote cannot trace; the
# compat backends receive a `Zygote.Buffer`-based out-of-place constraint
# closure instead (mirroring ADNLPModels' own Zygote integration).
function ADNLPModels.get_c(
    nlp::ADNLPModels.ADModel,
    ::Union{
        ZygoteJacobian,ZygoteJprod,ZygoteJtprod,ZygoteHessian,ZygoteHvprod,ZygoteGHjvprod
    },
)
    function c(x; nnln=nlp.meta.nnln)
        c = Zygote.Buffer(x, nnln)
        nlp.c!(c, x)
        return copy(c)
    end
    return c
end

end
