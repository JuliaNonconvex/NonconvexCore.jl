"""
    DictModel(obj = nothing)

An `AbstractModel` whose variables are named: each variable has a `Symbol` or
`String` key and arbitrary structured values, collected in an `OrderedDict`.
The JuMP model conversion is provided by the `NonconvexCoreJuMPExt` extension
when JuMP is loaded.
"""
mutable struct DictModel <: AbstractModel
    objective::Union{Nothing,Objective}
    eq_constraints::VectorOfFunctions
    ineq_constraints::VectorOfFunctions
    sd_constraints::VectorOfFunctions
    box_min::OrderedDict
    box_max::OrderedDict
    init::OrderedDict
    integer::OrderedDict
end

function DictModel(f=nothing)
    return DictModel(
        Objective(f),
        VectorOfFunctions(EqConstraint[]),
        VectorOfFunctions(IneqConstraint[]),
        VectorOfFunctions(SDConstraint[]),
        OrderedDict(),
        OrderedDict(),
        OrderedDict(),
        OrderedDict(),
    )
end

function addvar!(
    m::DictModel, k::Union{Symbol,String}, lb, ub; init=deepcopy(lb), integer=false
)
    getmin(m)[k] = lb
    getmax(m)[k] = ub
    m.init[k] = init
    m.integer[k] = integer
    return m
end

"""
    tovecmodel(m::DictModel, x0::AbstractVector)

A `DictModel` stores variables keyed by name, so an initial solution must be a
dict-like structure with the variable names as keys. This method throws a
descriptive error instead of failing deep inside the flattening machinery.
"""
function tovecmodel(m::DictModel, x0::AbstractVector)
    return error(
        "The initial solution of a DictModel must be a dict-like structure with the " *
        "variable names as keys. Got an AbstractVector of length $(length(x0)); use " *
        "Model instead of DictModel for vector-valued variables.",
    )
end
