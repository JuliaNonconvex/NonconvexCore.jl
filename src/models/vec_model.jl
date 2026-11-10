mutable struct VecModel{
    TO<:Union{Nothing,Objective},
    TE<:VectorOfFunctions,
    TI<:VectorOfFunctions,
    TS<:VectorOfFunctions,
    Tv1<:AbstractVector,
    Tv2<:AbstractVector,
    Tv3<:AbstractVector,
} <: AbstractModel
    objective::TO
    eq_constraints::TE
    ineq_constraints::TI
    sd_constraints::TS
    box_min::Tv1
    box_max::Tv2
    init::Tv3
    integer::BitVector
end

function isfeasible(model::VecModel, x::AbstractVector; ctol=1e-4)
    return all(getmin(model) .<= x .<= getmax(model)) &&
           all(getineqconstraints(model)(x) .<= ctol) &&
           all(-ctol .<= geteqconstraints(model)(x) .<= ctol)
end

function addvar!(m::VecModel, lb::Real, ub::Real; init::Real=lb, integer=false)
    push!(getmin(m), lb)
    push!(getmax(m), ub)
    push!(m.init, init)
    push!(m.integer, integer)
    return m
end
function addvar!(
    m::VecModel,
    lb::Vector{<:Real},
    ub::Vector{<:Real};
    init::Vector{<:Real}=copy(lb),
    integer=falses(length(lb)),
)
    append!(getmin(m), lb)
    append!(getmax(m), ub)
    append!(m.init, init)
    append!(m.integer, integer)
    return m
end

function getinit(m::VecModel)
    ma = getmax(m)
    mi = getmin(m)
    init = m.init
    return map(1:length(mi)) do i
        if isfinite(init[i])
            return init[i]
        else
            _ma = ma[i]
            _mi = mi[i]
            _ma == Inf && _mi == -Inf && return 0.0
            _ma == Inf && return _mi + 1.0
            _mi == -Inf && return _ma - 1.0
            return (_ma + _mi) / 2
        end
    end
end

get_objective_multiple(model::VecModel) = getobjective(model).multiple[]

function set_objective_multiple!(model::VecModel, m)
    getobjective(model).multiple[] = m
    return model
end

"""
Generic `optimize` for VecModel
"""
function optimize(model::VecModel, optimizer::AbstractOptimizer, x0, args...; kwargs...)
    workspace = Workspace(model, optimizer, copy(x0), args...; kwargs...)
    return optimize!(workspace)
end

"""
 Workspace constructor without x0
"""
function optimize(model::VecModel, optimizer::AbstractOptimizer, args...; kwargs...)
    workspace = Workspace(model, optimizer, args...; kwargs...)
    return optimize!(workspace)
end

"""
    FlattenedFunction

A callable that evaluates the wrapped function `f` on the structure rebuilt
by `unflattenx` from a flat vector, and flattens the output. The raw function
and the input unflattening are exposed via [`getrawfunction`](@ref NonconvexCore.getrawfunction) and
[`getinputunflatten`](@ref NonconvexCore.getinputunflatten) so that solvers can evaluate all model functions
against a single reconstructed structure per point.
"""
struct FlattenedFunction{F,U} <: Function
    f::F
    unflattenx::U
    flatteny::Bool
end
function (g::FlattenedFunction)(x)
    y = g.f(g.unflattenx(x)...)
    return g.flatteny ? maybeflatten(y)[1] : y
end

"""
    getrawfunction(f)

Returns the raw function wrapped by `f`, callable on the structure returned
by [`getinputunflatten`](@ref NonconvexCore.getinputunflatten). For a `FlattenedFunction` this is the
function as defined by the user before flattening.
"""
getrawfunction(f::FlattenedFunction) = f.f

"""
    getinputunflatten(f)

Returns the callable that rebuilds the user-facing input structure of the
function `f` from a flat vector.
"""
getinputunflatten(f::FlattenedFunction) = f.unflattenx

function tovecfunc(f, x::AbstractVector{<:Real}; flatteny=true)
    vx = float.(x)
    y = f(x)
    if y isa Real || y isa AbstractVector{<:Real}
        _flatteny = false
    else
        _flatteny = flatteny
    end
    if _flatteny
        tmp = maybeflatten(y)
        unflatteny = Unflatten(y, tmp[2])
        return first ∘ maybeflatten ∘ f, vx, unflatteny
    else
        return f, vx, identity
    end
end

function tovecfunc(f, x...; flatteny=true)
    vx, _unflattenx = flatten(x)
    unflattenx = Unflatten(x, _unflattenx)
    if flatteny
        y = f(x...)
        tmp = maybeflatten(y)
        # should be addressed in maybeflatten
        if y isa Real
            unflatteny = identity
        else
            unflatteny = Unflatten(y, tmp[2])
        end
        return FlattenedFunction(f, unflattenx, true), float.(vx), unflatteny
    else
        return FlattenedFunction(f, unflattenx, false), float.(vx), identity
    end
end

function tovecmodel(m::AbstractModel, _x0=getinit(m))
    x0 = reduce_type(_x0)
    box_min = reduce_type(m.box_min)
    box_max = reduce_type(m.box_max)
    init = reduce_type(m.init)
    v, _unflatten = flatten(x0)
    unflatten = Unflatten(x0, _unflatten)
    return VecModel(
        # objective
        Objective(tovecfunc(m.objective.f, x0)[1], m.objective.multiple, m.objective.flags),
        # eq_constraints
        if length(m.eq_constraints.fs) != 0
            VectorOfFunctions(
                map(Tuple(m.eq_constraints.fs)) do c
                    return EqConstraint(
                        tovecfunc(c.f, x0)[1], maybeflatten(c.rhs)[1], c.dim, c.flags
                    )
                end,
            )
        else
            VectorOfFunctions(())
        end,
        # ineq_constraints
        if length(m.ineq_constraints.fs) != 0
            VectorOfFunctions(
                map(Tuple(m.ineq_constraints.fs)) do c
                    return IneqConstraint(
                        tovecfunc(c.f, x0)[1], maybeflatten(c.rhs)[1], c.dim, c.flags
                    )
                end,
            )
        else
            VectorOfFunctions(())
        end,
        # sd_constraints
        if length(m.sd_constraints.fs) != 0
            VectorOfFunctions(
                map(Tuple(m.sd_constraints.fs)) do c
                    return SDConstraint(tovecfunc(c.f, x0; flatteny=false)[1], c.dim)
                end,
            )
        else
            VectorOfFunctions(())
        end,
        # box_min
        float.(flatten(box_min)[1]),
        # box_max
        float.(flatten(box_max)[1]),
        # init
        float.(flatten(init)[1]),
        # integer
        convert(BitVector, flatten(m.integer)[1]),
    ),
    float.(v),
    unflatten
end
