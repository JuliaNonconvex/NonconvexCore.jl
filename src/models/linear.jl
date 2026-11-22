"""
    LinearFunction(A, b = nothing)

A first-class linear (affine) function `x ↦ A x + b` (or `A x` when `b` is
`nothing`), with `getdim` equal to the number of rows of `A`. Recognized as
such by the model layer — solvers classify the corresponding constraints as
linear exactly and model transforms ([`nullspace_transform`](@ref),
[`unconstrained_transform`](@ref)) can act on them structurally.

Combine with a set for the function-in-set form:

```julia
add_linear_constraint!(m, A, LessThan(ub))     # A x ≤ ub
add_linear_constraint!(m, A, EqualTo(rhs))     # A x = rhs
```
"""
struct LinearFunction{A,B} <: AbstractFunction
    A::A
    b::B
end
LinearFunction(A) = LinearFunction(A, nothing)
function (f::LinearFunction)(x::AbstractVector)
    out = f.A * x
    f.b === nothing || (out .+= f.b)
    return out
end
getdim(f::LinearFunction) = size(f.A, 1)

# The offset part of a LinearFunction (the zero vector of the right length
# when there is none).
linear_offset(f::LinearFunction) = f.b === nothing ? zeros(eltype(f.A), size(f.A, 1)) : f.b

"""
    add_linear_constraint!(m::AbstractModel, A, set::AbstractSet; b = nothing, flags = Set{Symbol}())

Adds the first-class linear constraint `LinearFunction(A, b) ∈ set` to the
model `m` in place, e.g. `A x ≤ ub` with `LessThan(ub)`, `A x = rhs` with
`EqualTo(rhs)` or `A x ⪰ 0` with `PSDCone(k)` (each row block of `A x`
reshaped; `PSDCone` is provided by NonconvexSemidefinite.jl). `flags` labels
the constraint for meta-algorithm selection. See [`LinearFunction`](@ref).
"""
function add_linear_constraint!(
    m::AbstractModel, A::AbstractMatrix, set::AbstractSet; b=nothing, flags=Set{Symbol}()
)
    return add_set_constraint!(m, FunctionInSet(LinearFunction(A, b), set; flags=flags))
end
