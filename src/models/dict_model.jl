mutable struct DictModel <: AbstractModel
    objective::Union{Nothing, Objective}
    eq_constraints::VectorOfFunctions
    ineq_constraints::VectorOfFunctions
    sd_constraints::VectorOfFunctions
    box_min::OrderedDict
    box_max::OrderedDict
    init::OrderedDict
    integer::OrderedDict
    adbackend::AD.AbstractBackend
end

function DictModel(f = nothing; adbackend = AD.ZygoteBackend())
    return DictModel(
        Objective(f), VectorOfFunctions(EqConstraint[]),
        VectorOfFunctions(IneqConstraint[]),
        VectorOfFunctions(SDConstraint[]),
        OrderedDict(), OrderedDict(),
        OrderedDict(), OrderedDict(), adbackend,
    )
end

function addvar!(m::DictModel, k::Union{Symbol, String}, lb, ub; init = deepcopy(lb), integer = false)
    getmin(m)[k] = lb
    getmax(m)[k] = ub
    m.init[k] = init
    m.integer[k] = integer
    return m
end
