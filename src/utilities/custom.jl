import Zygote: unthunk

"""
    CustomGradFunction(f, g)

Wraps the function `f` so that its gradient (or Jacobian, for vector-valued
`f`) is computed by the user-supplied function `g` instead of automatic
differentiation.
"""
struct CustomGradFunction{F,G} <: Function
    f::F
    g::G
end
(f::CustomGradFunction)(x::AbstractVector) = f.f(x)
function ChainRulesCore.rrule(f::CustomGradFunction, x::AbstractVector)
    v = f.f(x)
    if f.g === nothing
        if v isa Real
            G = spzeros(eltype(v), length(x))
        else
            G = spzeros(eltype(v), length(v), length(x))
        end
    else
        G = f.g(x)
    end
    return v, Δ -> begin
        Δv = unthunk(Δ)
        if G isa AbstractVector
            return (NoTangent(), G * Δv)
        elseif G isa LazyJacobian
            return (NoTangent(), G' * Δv)
        elseif G isa Function
            # an HVP operator (x-curried): G(Δv) applies the second derivative
            return (NoTangent(), G(Δv))
        else
            spΔ = dropzeros!(sparse(Δv))
            if length(spΔ.nzval) == 1
                return (NoTangent(), G[spΔ.nzind[1], :] * spΔ.nzval[1])
            else
                return (NoTangent(), G' * Δv)
            end
        end
    end
end
function ChainRulesCore.frule(Δs::Tuple, f::CustomGradFunction, x::AbstractVector)
    _, Δx = Δs
    v = f.f(x)
    if f.g === nothing
        if v isa Real
            ∇ = spzeros(eltype(v), 1, length(x))
        else
            ∇ = spzeros(eltype(v), length(v), length(x))
        end
    else
        ∇ = f.g(x)
    end
    project_to = ProjectTo(v)
    if ∇ isa Function
        # an HVP operator: ∇(Δx) is the directional derivative of v
        return v, project_to(∇(Δx))
    elseif ∇ isa AbstractVector && Δx isa AbstractVector
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v, project_to(zero(eltype(Δx)))
        else
            return v, project_to(∇' * Δx)
        end
    else
        if !(∇ isa LazyJacobian) && issparse(∇) && nnz(∇) == 0
            return v,
            project_to(v isa Real ? zero(eltype(v)) : spzeros(eltype(Δx), size(∇, 1)))
        elseif v isa Real
            return v, project_to(sum(∇ * Δx))
        else
            return v, project_to(_sparse_reshape(∇ * Δx, size(v)...))
        end
    end
end
@ForwardDiff_frule (f::CustomGradFunction)(x::AbstractVector{<:ForwardDiff.Dual})

"""
    CustomHessianFunction(f, g, h; hvp = false)

Wraps the function `f` so that its gradient (or Jacobian, for vector-valued
`f`) is computed by `g` and its Hessian by `h` (or, with `hvp = true`, by a
Hessian-vector product operator).

For a scalar-valued `f`, `h(x)` returns the Hessian matrix (or, with
`hvp = true`, `h(x, v)` returns the Hessian-vector product `∇²f(x)·v`).

For a vector-valued `f` (e.g. a block equality constraint), pass
`hvp = true` and let `h(x, v)` return the per-component Hessian-vector
products: an array shaped like `J(x) * v` whose `i`-th row is
`∇²fᵢ(x)·v`. The weighted Lagrangian Hessian `Σ μᵢ∇²fᵢ(x)` is then
assembled from those products, which is all the second-order solvers
need from a vector-valued constraint.
"""
struct CustomHessianFunction{F,G,H} <: Function
    f::F
    g::G
    h::H
    function CustomHessianFunction(f::F, g::G, h::H; hvp=false) where {F,G,H}
        _h = hvp ? (x -> (v -> h(x, v))) : h
        return new{F,G,typeof(_h)}(f, g, _h)
    end
end
(to::CustomHessianFunction)(x) = to.f(x)
function ChainRulesCore.rrule(f::CustomHessianFunction, x)
    v = f(x)
    g = CustomGradFunction(f.g, f.h)
    G = g(x)
    return v, Δ -> begin
        Δv = unthunk(Δ)
        if v isa Real
            return (NoTangent(), G * Δv)
        else
            return (NoTangent(), G' * Δv)
        end
    end
end
function ChainRulesCore.frule(Δs::Tuple, f::CustomHessianFunction, x::AbstractVector)
    _, Δx = Δs
    v = f(x)
    project_to = ProjectTo(v)
    if v isa Real
        g = CustomGradFunction(f.g, f.h)
        ∇ = g(x)
        if ∇ isa AbstractVector && Δx isa AbstractVector
            return v, project_to(∇' * unthunk(Δx))
        else
            return v, project_to(∇ * unthunk(Δx))
        end
    else
        # vector-valued: the Jacobian gives the first derivative; the
        # second-order replay differentiates the pullback through the
        # wrapper's custom HVP operator
        J = f.g(x)
        return v, project_to(J * unthunk(Δx))
    end
end
@ForwardDiff_frule (f::CustomHessianFunction)(x::AbstractVector{<:ForwardDiff.Dual})
