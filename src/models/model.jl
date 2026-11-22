"""
```
struct Model <: AbstractModel
    objective::Objective
    eq_constraints::VectorOfFunctions
    ineq_constraints::VectorOfFunctions
    box_min::AbstractVector
    box_max::AbstractVector
    init::AbstractVector
    integer::AbstractVector
end
```

The `Model` structs stores information about the nonlinear optimization problem.
- `objective`: the objective function of the problem of type [`Objective`](@ref NonconvexCore.Objective).
- `eq_constraints`: the equality constraints of the problem of type [`VectorOfFunctions`](@ref NonconvexCore.VectorOfFunctions). Each function in `ineq_constraints` can be an instance of [`IneqConstraint`](@ref NonconvexCore.IneqConstraint) or [`AbstractFunction`](@ref NonconvexCore.AbstractFunction). If the function is not an `IneqConstraint`, its right-hand-side bound is assumed to be 0.
- `ineq_constraints`: the inequality constraints of the problem of type [`VectorOfFunctions`](@ref NonconvexCore.VectorOfFunctions). Each function in `ineq_constraints` can be an instance of [`IneqConstraint`](@ref NonconvexCore.IneqConstraint) or [`AbstractFunction`](@ref NonconvexCore.AbstractFunction). If the function is not an `IneqConstraint`, its right-hand-side bound is assumed to be 0.
- `sd_constraints`: a list of matrix-valued functions which must be positive semidefinite at any feasible solution.
"""
mutable struct Model{Tv<:AbstractVector} <: AbstractModel
    objective::Union{Nothing,Objective}
    eq_constraints::VectorOfFunctions
    ineq_constraints::VectorOfFunctions
    sd_constraints::VectorOfFunctions
    set_constraints::VectorOfFunctions
    set_variables::Vector{VariableInSet}
    box_min::Tv
    box_max::Tv
    init::Tv
    integer::AbstractVector
end

# Backward-compatible constructor: the `set_constraints`/`set_variables`
# fields were added in NonconvexCore 2.3; older callers pass 8 fields.
function Model(obj, eq, ineq, sd, box_min, box_max, init, integer)
    return Model(
        obj,
        eq,
        ineq,
        sd,
        VectorOfFunctions(FunctionInSet[]),
        VariableInSet[],
        box_min,
        box_max,
        init,
        integer,
    )
end

"""
    Model()
    Model(f)

Constructs an empty model or a model with objective function `f`. The decision variables are assumed to be of type `Vector{Any}`.
"""
Model(f::Union{Nothing,Function}=x -> 0.0) = Model(Any, f)

"""
    Model(::Type{T}, f::Union{Nothing, Function}) where {T}

Constructs an empty model with objective function `f` and decision variable value type `Vector{T}`. `f` can be an instance of `Base.Function` but must return a number,  or it can be an intance of [`Objective`](@ref NonconvexCore.Objective). `f` can also be `nothing` in which case, the objective function can be defined later using [`set_objective!`](@ref).
"""
Model(::Type{T}, f::Function) where {T} = Model(T, Objective(f))
function Model(::Type{T}, obj::Union{Nothing,Objective}) where {T}
    return Model(
        obj,
        VectorOfFunctions(EqConstraint[]),
        VectorOfFunctions(IneqConstraint[]),
        VectorOfFunctions(SDConstraint[]),
        VectorOfFunctions(FunctionInSet[]),
        VariableInSet[],
        T[],
        T[],
        T[],
        [],
    )
end

"""
    getobjective(m::AbstractModel)

Returns the `Objective` of the model `m`, or `nothing` if no objective has
been set.
"""
getobjective(m::AbstractModel) = m.objective

getineqconstraints(m::AbstractModel) = m.ineq_constraints

geteqconstraints(m::AbstractModel) = m.eq_constraints

getsdconstraints(m::AbstractModel) = m.sd_constraints

"""
    getsetconstraints(m::AbstractModel)

Returns the function-in-set constraints ([`FunctionInSet`](@ref)) of the
model `m`.
"""
getsetconstraints(m::AbstractModel) = m.set_constraints

"""
    getsetvariables(m::AbstractModel)

Returns the variable-in-set blocks ([`VariableInSet`](@ref)) of the model
`m`, in creation order. Plain `addvar!` blocks are registered with the box
set `Interval(lb, ub)`.
"""
getsetvariables(m::AbstractModel) = m.set_variables

function getobjectiveconstraints(m::AbstractModel)
    return VectorOfFunctions((getobjective(m), getineqconstraints(m), geteqconstraints(m)))
end

getineqconstraint(m::AbstractModel, i::Integer) = getineqconstraints(m).fs[i]

geteqconstraint(m::AbstractModel, i::Integer) = geteqconstraints(m).fs[i]

getnineqconstraints(m::AbstractModel) = length(getineqconstraints(m))

getneqconstraints(m::AbstractModel) = length(geteqconstraints(m))

getnsdconstraints(m::AbstractModel) = length(getsdconstraints(m))

getnsetconstraints(m::AbstractModel) = length(getsetconstraints(m).fs)

function getnconstraints(m::AbstractModel)
    return getnineqconstraints(m) +
           getneqconstraints(m) +
           getnsdconstraints(m) +
           getnsetconstraints(m)
end

getnvars(m::AbstractModel) = length(getmin(m))

"""
    getdim(m::AbstractModel)

Returns a 2-tuple of the number of constraints and the number of variables in the model `m`.
"""
getdim(m::AbstractModel) = (getnconstraints(m), getnvars(m))

function reduce_type(d::AbstractVector)
    return reduce_type.(d)
end
function reduce_type(d::OrderedDict)
    return OrderedDict(identity.(keys(d)) .=> reduce_type.(values(d)))
end
reduce_type(d) = d

"""
    getmin(m::AbstractModel)
    getmin(m::AbstractModel, i)

Returns the vector of lower bounds of the decision variables of the model
`m`, or the lower bound of the `i`-th variable.
"""
getmin(m::AbstractModel) = m.box_min
getmin(m::AbstractModel, i) = getmin(m)[i]
Base.isinteger(m::AbstractModel, i) = m.integer[i]

"""
    getmax(m::AbstractModel)
    getmax(m::AbstractModel, i)

Returns the vector of upper bounds of the decision variables of the model
`m`, or the upper bound of the `i`-th variable.
"""
getmax(m::AbstractModel) = m.box_max
getmax(m::AbstractModel, i) = getmax(m)[i]

getinit(model::AbstractModel) = model.init

"""
    setinteger!(m::AbstractModel, i, integer)

Marks the `i`-th decision variable of the model `m` as integer (`integer =
true`) or continuous (`integer = false`). Integer variables are only
supported by a subset of the optimizers; see their
[`solver_capabilities`](@ref NonconvexCore.solver_capabilities).
"""
function setinteger!(m::AbstractModel, i, integer)
    m.integer[i] = integer
    return m
end

"""
    addvar!(m::Model, lb, ub; init=deepcopy(lb), integer=false)

Adds a decision variable with lower bound(s) `lb` and upper bound(s) `ub` to
the model `m` in place, optionally with an initial value `init` and an
`integer` flag. `lb` and `ub` can be scalars or vectors. With `Model`
(packed variables) a variable is a single scalar; use `lb`/`ub` vectors to
add several at once, and `DictModel` to add named variables.

Under the hood the variable block is registered as a variable in the box set
`Interval(lb, ub)` (a [`VectorSet`](@ref)); see [`getsetvariables`](@ref) and
[`add_set_variable!`](@ref) for the general set-based form.
"""
function addvar!(
    m::Model, lb::Real, ub::Real; init=deepcopy(lb), integer=false, flags=Set{Symbol}()
)
    push!(getmin(m), lb)
    push!(getmax(m), ub)
    push!(m.init, init)
    push!(m.integer, integer)
    push!(
        m.set_variables,
        VariableInSet(Interval(lb, ub), length(m.box_min):length(m.box_min); flags=flags),
    )
    return m
end
function addvar!(
    m::Model,
    lb::Vector,
    ub::Vector;
    init=deepcopy(lb),
    integer=falses(sum(length, lb)),
    flags=Set{Symbol}(),
)
    first = length(m.box_min) + 1
    append!(getmin(m), lb)
    append!(getmax(m), ub)
    append!(m.init, init)
    append!(m.integer, integer)
    push!(
        m.set_variables,
        VariableInSet(Interval(lb, ub), first:(first + length(lb) - 1); flags=flags),
    )
    return m
end
# Generic fallback (back-compat): any other bound pair is recorded as a
# single variable, e.g. a whole struct or NamedTuple with `Model(f)` over
# `Vector{Any}`; the bounds are compared/scattered by the flattening.
function addvar!(m::Model, lb, ub; init=deepcopy(lb), integer=false, flags=Set{Symbol}())
    push!(getmin(m), lb)
    push!(getmax(m), ub)
    push!(m.init, init)
    push!(m.integer, integer)
    push!(
        m.set_variables,
        VariableInSet(
            Interval(promote(lb, ub)..., 1),
            length(m.box_min):length(m.box_min);
            flags=flags,
        ),
    )
    return m
end

"""
    add_set_variable!(m::Model, set::AbstractSet; init = nothing, integer = false)

Adds a block of decision variables constrained to lie in the set `set`, in
place. The block is registered as a [`VariableInSet`](@ref) (returned):
`PSDCone(n)` introduces an `n × n`
semidefinite matrix variable, `Simplex(n)` a probability simplex variable,
the order sets bounded/fixed blocks, and `SetProduct` a concatenation of
blocks. The cone parts the box cannot express (PSD-ness, the simplex sum)
become synthetic set constraints at model conversion, so solvers handle
variable sets exactly like [`FunctionInSet`](@ref) constraints — declare the
`sets` and `variable_sets` capabilities.
"""
function add_set_variable!(
    m::Model, set::AbstractSet; init=nothing, integer=false, flags=Set{Symbol}()
)
    lb, ub = variable_box(set)
    n = length(lb)
    x0 = init === nothing ? variable_init(set) : init
    length(x0) == n || throw(
        ArgumentError(
            "The initial point of a $(typeof(set)) block must have $n entries, got $(length(x0)).",
        ),
    )
    first = length(m.box_min) + 1
    append!(getmin(m), lb)
    append!(getmax(m), ub)
    append!(m.init, collect(x0))
    append!(m.integer, fill(integer, n))
    push!(m.set_variables, VariableInSet(set, first:(first + n - 1); flags=flags))
    return m.set_variables[end]
end

"""
    set_objective!(m::AbstractModel, f::Function)

Sets the objective of the moodel `m` to the function `f`. `f` must return a scalar.
"""
function set_objective(m::AbstractModel, f::Function; kwargs...)
    newm = @set m.objective = Objective(f; kwargs...)
    return newm
end

"""
    set_objective!(m::AbstractModel, f::Function; kwargs...)

Sets the objective of the model `m` to the function `f` in place. Keyword
arguments are forwarded to the [`Objective`](@ref NonconvexCore.Objective)
constructor. See also [`set_objective`](@ref NonconvexCore.set_objective)
for a non-mutating variant.
"""
function set_objective!(m::AbstractModel, f::Function; kwargs...)
    m.objective = Objective(f; kwargs...)
    return m
end

"""
    add_ineq_constraint!(m::AbstractModel, f, s=0.0; kwargs...)
    add_ineq_constraint!(m::AbstractModel, f::AbstractFunction, s=0.0; kwargs...)
    add_ineq_constraint!(m::AbstractModel, f::IneqConstraint)
    add_ineq_constraint!(m::AbstractModel, fs::Vector{<:IneqConstraint})

Adds the inequality constraint `f(x) ≤ s` to the model `m` in place.
`f` can be a plain function (a `FunctionWrapper` is constructed
automatically) or a constraint object. Several constraints can be appended
at once with a vector of `IneqConstraint`s.
"""
function add_ineq_constraint!(
    m::AbstractModel,
    f::Function,
    s=0.0;
    dim=length(flatten(f(reduce_type(getinit(m))))[1]),
    kwargs...,
)
    return add_ineq_constraint!(m, FunctionWrapper(f, dim), s; kwargs...)
end
function add_ineq_constraint!(m::AbstractModel, f::AbstractFunction, s=0.0; kwargs...)
    return add_ineq_constraint!(m, IneqConstraint(f, s; kwargs...))
end
function add_ineq_constraint!(m::AbstractModel, f::IneqConstraint)
    push!(m.ineq_constraints.fs, f)
    return m
end
function add_ineq_constraint!(m::AbstractModel, fs::Vector{<:IneqConstraint})
    append!(m.ineq_constraints.fs, fs)
    return m
end

"""
    add_eq_constraint!(m::AbstractModel, f, s=0.0; kwargs...)
    add_eq_constraint!(m::AbstractModel, f::AbstractFunction, s=0.0; kwargs...)
    add_eq_constraint!(m::AbstractModel, f::EqConstraint)
    add_eq_constraint!(m::AbstractModel, fs::Vector{<:EqConstraint})

Adds the equality constraint `f(x) = s` to the model `m` in place.
`f` can be a plain function (a `FunctionWrapper` is constructed
automatically) or a constraint object. Several constraints can be appended
at once with a vector of `EqConstraint`s.
"""
function add_eq_constraint!(
    m::AbstractModel,
    f::Function,
    s=0.0;
    dim=length(flatten(f(reduce_type(getinit(m))))[1]),
    kwargs...,
)
    return add_eq_constraint!(m, FunctionWrapper(f, dim), s; kwargs...)
end
function add_eq_constraint!(m::AbstractModel, f::AbstractFunction, s=0.0; kwargs...)
    return add_eq_constraint!(m, EqConstraint(f, s; kwargs...))
end
function add_eq_constraint!(m::AbstractModel, f::EqConstraint)
    push!(m.eq_constraints.fs, f)
    return m
end
function add_eq_constraint!(m::AbstractModel, fs::Vector{<:EqConstraint})
    append!(m.eq_constraints.fs, fs)
    return m
end

"""
    add_sd_constraint!(m::AbstractModel, f, dim=size(f(...), 1))
    add_sd_constraint!(m::AbstractModel, sd_constraint::AbstractFunction)

Adds the semidefinite constraint `f(x) ⪰ 0` (the matrix-valued function
`f(x)` must be positive semidefinite) to the model `m` in place. The
matrix dimension `dim` is inferred when `f` is a plain function.
"""
function add_sd_constraint!(
    m::AbstractModel, f::Function, dim=size(f(reduce_type(getinit(m))), 1)
)
    return add_sd_constraint!(m, SDConstraint(f, dim))
end
function add_sd_constraint!(m::AbstractModel, sd_constraint::AbstractFunction)
    push!(m.sd_constraints.fs, sd_constraint)
    return m
end

"""
    add_set_constraint!(m::AbstractModel, f::Function, set::AbstractSet)
    add_set_constraint!(m::AbstractModel, c::FunctionInSet)

Adds the function-in-set constraint `f(x) ∈ set` to the model `m` in place.
`f` must return a value whose shape matches `set`: a scalar or vector for the
order sets, a matrix for a semidefinite cone (`PSDCone`, provided by
NonconvexSemidefinite.jl), or a tuple with one entry per component of a
[`SetProduct`](@ref). The output shape is validated by evaluating `f` once at
the model's initial point.

Solvers must declare support for function-in-set constraints via the `sets`
capability of [`ConstraintCapabilities`](@ref); `AugLagAlg` from
NonconvexAugLagLab is the reference implementation.
"""
function add_set_constraint!(
    m::AbstractModel, f::Function, set::AbstractSet; flags=Set{Symbol}()
)
    return add_set_constraint!(m, FunctionInSet(f, set; flags=flags))
end
function add_set_constraint!(m::AbstractModel, c::FunctionInSet)
    # Validate the output shape once at the initial point (`(c)(x)` checks).
    c(reduce_type(getinit(m)))
    push!(m.set_constraints.fs, c)
    return m
end

"""
    add_chance_constraint!(m::AbstractModel, c::AbstractConstraint, scenarios;
        epsilon = 0.05, kappa = 10.0)
    add_chance_constraint!(m::AbstractModel, f, scenarios, set::AbstractSet;
        epsilon = 0.05, kappa = 10.0, flags = Set{Symbol}())

Adds the probabilistic constraint
`P_ξ[f(x, ξ) ∈ set] ≥ 1 - epsilon` to the model `m` in place. Any existing
constraint type can be turned probabilistic: pass a classical constraint
(`IneqConstraint`, `EqConstraint`, `SDConstraint`) or a `FunctionInSet` whose
function takes the decision vector `x` and a random scenario `ξ`; the bound
is the constraint's own set. Equivalently, pass the function and the bound
directly as a set (`LessThan`, `Interval`, `PSDCone` from
NonconvexSemidefinite.jl, …) together with the scenario batch `scenarios`
(a vector of `ξ` values drawn by the caller with the RNG of its choice —
NonconvexCore does not sample).

The exceedance probability is estimated by a smooth (tanh) sample average
of the indicator with sharpness `kappa` (see [`ChanceSet`](@ref)), making
the constraint differentiable. With a `SetProduct` bound, `epsilon` can be
a vector of probabilities, one per output, each output sharing the same
scenario batch; `epsilon = Zero()` expresses a deterministic constraint.
Requires the `sets_chance` solver capability.
"""
function add_chance_constraint!(
    m::AbstractModel, c::AbstractConstraint, scenarios; flags=Set{Symbol}(), kwargs...
)
    return add_set_constraint!(m, chance(c, scenarios; kwargs...))
end
function add_chance_constraint!(
    m::AbstractModel,
    f::Function,
    scenarios,
    set::AbstractSet;
    epsilon=0.05,
    kappa=10.0,
    flags=Set{Symbol}(),
)
    return add_set_constraint!(
        m,
        FunctionInSet(
            f, ChanceSet(set, scenarios; epsilon=epsilon, kappa=kappa); flags=flags
        ),
    )
end

"""
    add_epigraph_variable!(m::AbstractModel; margin = 1.0, t_lb = -Inf, t_ub = Inf)

Reformulates the model in place with the epigraph trick: a new scalar
variable `t` (appended to the variables, initialized at
`objective(init) + margin`) becomes the objective, and the former objective
`f` is dropped from the objective slot so it can be constrained by the
caller — typically inside a [`SetProduct`](@ref) component `f(x) - t`, so
that one function evaluation feeds the objective bound and other
constraints at once. Returns the [`VariableInSet`](@ref) block of `t`.

[`epigraph!`](@ref) is the fully automatic variant that also adds the
epigraph inequality `f(x) - t ≤ 0` as a set constraint.
"""
function add_epigraph_variable!(m::Model; margin=1.0, t_lb=(-Inf), t_ub=Inf, integer=false)
    obj = m.objective
    obj isa Objective ||
        throw(ArgumentError("The model has no objective function to move to the epigraph."))
    f = getfunction(obj)
    n_old = length(getmin(m))
    t0 = f(reduce_type(getinit(m))) + margin
    block = add_set_variable!(m, Interval(t_lb, t_ub); init=[t0], integer=integer)
    m.objective = Objective(x -> x[end], obj.multiple, obj.flags)
    return block, f, n_old
end

"""
    epigraph!(m::AbstractModel; margin = 1.0, t_lb = -Inf, t_ub = Inf)

Reformulates the model in place into its epigraph form: the new variable `t`
becomes the objective and the former objective `f` moves into the set
constraint `f(x) - t ∈ LessThan(0.0)` (`f(x) ≤ t`). Minimizing `t` then
minimizes `f`, and the objective bound participates in the model's
[`FunctionInSet`](@ref) machinery like any other constraint. Returns the
[`VariableInSet`](@ref) block of `t`.
"""
function epigraph!(m::Model; margin=1.0, t_lb=(-Inf), t_ub=Inf, integer=false)
    block, f, n_old = add_epigraph_variable!(m; margin=margin, t_lb=t_lb, t_ub=t_ub)
    add_set_constraint!(m, FunctionInSet(x -> f(view(x, 1:n_old)) - x[end], LessThan(0.0)))
    return block
end

function epigraph!(m::AbstractModel; kwargs...)
    return throw(
        ArgumentError(
            "epigraph! is implemented for `Model` (vector variables). Got " *
            "$(typeof(m)); convert the model first or use add_epigraph_variable! " *
            "and add the epigraph inequality with add_set_constraint!.",
        ),
    )
end
