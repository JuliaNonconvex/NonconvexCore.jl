module NonconvexCore

const debugging = Ref(false)
const show_residuals = Ref(false)

export  Model,
        DictModel,
        addvar!,
        getobjective,
        set_objective!,
        add_ineq_constraint!,
        add_eq_constraint!,
        add_sd_constraint!,
        decompress_symmetric,
        getmin,
        getmax,
        setmin!,
        setmax!,
        setinteger!,
        optimize,
        Workspace,
        KKTCriteria,
        IpoptCriteria,
        FunctionWrapper,
        Tolerance,
        GenericCriteria,
        KKTCriteria,
        ScaledKKTCriteria,
        IpoptCriteria

using Parameters, Zygote, ChainRulesCore, ForwardDiff, Requires
using SparseArrays, Reexport
using NamedTupleTools
@reexport using LinearAlgebra, OrderedCollections
using Reexport, Setfield
import JuMP, MathOptInterface
const MOI = MathOptInterface
using SolverCore: log_header, log_row
using JuMP: VariableRef, is_binary, is_integer, has_lower_bound,
            has_upper_bound, lower_bound, upper_bound,
            start_value, ConstraintRef, constraint_object,
            AffExpr, objective_function, objective_sense

# General

include("utilities/params.jl")
include("functions/functions.jl")
include("functions/value_jacobian.jl")
include("functions/counting_function.jl")
include("functions/sparse.jl")
include("common.jl")
include("utilities/callbacks.jl")
include("utilities/convergence.jl")

# Models

include("models/flatten.jl")
include("models/model.jl")
include("models/vec_model.jl")
include("models/dict_model.jl")
include("models/model_docs.jl")
include("models/jump.jl")
include("models/moi.jl")

end
