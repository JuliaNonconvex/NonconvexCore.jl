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
function ZygoteJacobian(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    # JuMP-converted models: exact Float64 derivatives straight from the
    # MOI evaluator (the Zygote Jacobian can miss the JuMPModel* rrules
    # through the Buffer-based constraint closure).
    jump_evaluator isa JuMPEvaluator && return JuMPJacBackend(jump_evaluator)
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
function ZygoteHessian(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    # JuMP-converted models: the exact evaluator Hessian replaces the
    # forward-over-reverse (FD-of-exact-gradients) approximation.
    jump_evaluator isa JuMPEvaluator &&
        return JuMPHessBackend(jump_evaluator, _jump_hess_positions(jump_evaluator, nvar))
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
function ZygoteHvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    # JuMP-converted models: the forward-over-forward path pushes Duals
    # into the Float64-only MOI evaluator and crashes; the exact evaluator
    # Hessian-vector product is Float64-safe.
    jump_evaluator isa JuMPEvaluator && return JuMPHvBackend(jump_evaluator)
    return ZygoteHvprod()
end
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

# ---------------------------------------------------------------------------
# Exact evaluator backends for JuMP-converted models
# ---------------------------------------------------------------------------
# The JuMPModel* rrules live on the vector-level callables; the Zygote paths
# above do not always see them (the Buffer-based constraint closure makes
# Zygote.jacobian miss the rrules and produce a wrong Jacobian, and the
# ForwardDiff-based Hvprod! pushes Duals into the Float64-only MOI evaluator
# and crashes). These backends read the derivatives straight from
# NonconvexCore's JuMPEvaluator (`jump_jacobian!`,
# `jump_hessian_lagrangian!`): exact and Float64-safe.
using NLPModels: NLPModels
using NonconvexCore: JuMPEvaluator, jump_jacobian!, jump_hessian_lagrangian!

struct JuMPJacBackend{M} <: ADBackend
    m::M
    JuMPJacBackend(m::M) where {M} = new{M}(m)
end
function JuMPJacBackend(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    jump_evaluator isa JuMPEvaluator || throw(
        ArgumentError("JuMPJacBackend requires a JuMPEvaluator (pass `jump_evaluator`)")
    )
    return JuMPJacBackend(jump_evaluator)
end
function ADNLPModels.get_nln_nnzj(b::JuMPJacBackend, nvar, ncon)
    return length(b.m.jac_rows)
end
function NLPModels.jac_structure!(
    b::JuMPJacBackend, nlp::ADNLPModels.ADModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}
)
    copyto!(rows, b.m.jac_rows)
    copyto!(cols, b.m.jac_cols)
    return rows, cols
end
function NLPModels.jac_coord!(b::JuMPJacBackend, nlp::ADNLPModels.ADModel, x::AbstractVector, vals::AbstractVector)
    jump_jacobian!(b.m, vals, x)
    return vals
end

struct JuMPHvBackend{M} <: ADBackend
    m::M
    JuMPHvBackend(m::M) where {M} = new{M}(m)
end
function JuMPHvBackend(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    jump_evaluator isa JuMPEvaluator || throw(
        ArgumentError("JuMPHvBackend requires a JuMPEvaluator (pass `jump_evaluator`)")
    )
    return JuMPHvBackend(jump_evaluator)
end
# The objective weight (and, for `:lag`, the multipliers) are folded into the
# Lagrangian closure `ℓ` by ADNLPModels before Hvprod! is called; the
# evaluator needs them explicitly, so they are passed again (obj_weight / y)
# and `ℓ` is ignored.
function ADNLPModels.Hvprod!(b::JuMPHvBackend, Hv, x, v, ℓ, ::Val{:obj}, obj_weight)
    return _jump_hv!(b.m, Hv, x, v, obj_weight, zeros(b.m.nineq + b.m.neq))
end
function ADNLPModels.Hvprod!(b::JuMPHvBackend, Hv, x, v, ℓ, ::Val{:lag}, y, obj_weight)
    return _jump_hv!(b.m, Hv, x, v, obj_weight, y)
end
function ADNLPModels.Hvprod!(b::JuMPHvBackend, Hv, x, v, ci, ::Val{:ci})
    λ = zeros(b.m.nineq + b.m.neq)
    λ[ci] = 1.0
    return _jump_hv!(b.m, Hv, x, v, 0.0, λ)
end
function _jump_hv!(m, Hv, x, v, obj_factor, lambda)
    sv = zeros(length(m.hess_rows))
    jump_hessian_lagrangian!(m, sv, x, obj_factor, lambda)
    fill!(Hv, zero(eltype(Hv)))
    @inbounds for k in eachindex(m.hess_rows)
        i = m.hess_rows[k]
        j = m.hess_cols[k]
        w = sv[k]
        Hv[i] += w * v[j]
        i == j || (Hv[j] += w * v[i])
    end
    return Hv
end

struct JuMPHessBackend{M} <: ADBackend
    m::M
    pos::Vector{Int}   # position of each sparse entry in the dense lower triangle
    JuMPHessBackend(m::M, pos::Vector{Int}) where {M} = new{M}(m, pos)
end
# dense lower-triangle position of each sparse Hessian entry (i, j), i >= j,
# in the column-major ordering `for j in 1:n, i in j:n`
function _jump_hess_positions(m, nvar)
    pos = Int[]
    for k in eachindex(m.hess_rows)
        i = m.hess_rows[k]
        j = m.hess_cols[k]
        push!(pos, (j - 1) * nvar - div((j - 1) * (j - 2), 2) + (i - j + 1))
    end
    return pos
end

function JuMPHessBackend(nvar, f, ncon=0, c!::Function=(args...) -> []; jump_evaluator=nothing, kwargs...)
    jump_evaluator isa JuMPEvaluator || throw(
        ArgumentError("JuMPHessBackend requires a JuMPEvaluator (pass `jump_evaluator`)")
    )
    return JuMPHessBackend(jump_evaluator, _jump_hess_positions(jump_evaluator, nvar))
end
function NLPModels.hess_coord!(
    b::JuMPHessBackend,
    nlp::ADNLPModels.ADModel,
    x::AbstractVector,
    y::AbstractVector,
    obj_weight::Real,
    vals::AbstractVector,
)
    return _jump_hess_coord!(b, vals, x, obj_weight, y)
end
function NLPModels.hess_coord!(
    b::JuMPHessBackend,
    nlp::ADNLPModels.ADModel,
    x::AbstractVector,
    obj_weight::AbstractFloat,
    vals::AbstractVector,
)
    return _jump_hess_coord!(b, vals, x, obj_weight, zeros(b.m.nineq + b.m.neq))
end
function _jump_hess_coord!(b, vals, x, obj_factor, lambda)
    sv = zeros(length(b.m.hess_rows))
    jump_hessian_lagrangian!(b.m, sv, x, obj_factor, lambda)
    fill!(vals, zero(eltype(vals)))
    @inbounds for k in eachindex(b.m.hess_rows)
        vals[b.pos[k]] += sv[k]
    end
    return vals
end
end
