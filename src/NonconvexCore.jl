module NonconvexCore

const debugging = Ref(false)
const show_residuals = Ref(false)

export Model,
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
    IpoptCriteria,
    ConstraintCapabilities,
    merge_capabilities,
    solver_capabilities,
    checksubproblem!,
    set_ad_backend!,
    get_ad_backend

using Parameters, Zygote, ChainRulesCore, ForwardDiff
using SparseArrays, Reexport, Requires
@reexport using LinearAlgebra, OrderedCollections
@reexport using ADTypes
import DifferentiationInterface as DI
using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity, hessian_sparsity
using Reexport, Setfield

"""
    DEFAULT_AD_BACKEND

The automatic differentiation backend used by default wherever NonconvexCore
computes derivatives, e.g. in `value_gradient`, `value_jacobian`,
`sparse_jacobian` and the solver-facing derivative evaluators. It is initially
`ADTypes.AutoZygote()`. Change it with [`set_ad_backend!`](@ref), or pass a
backend explicitly to the functions above.
"""
const DEFAULT_AD_BACKEND = Ref{AbstractADType}(AutoZygote())

"""
    set_ad_backend!(backend::ADTypes.AbstractADType)

Sets the package-wide default automatic differentiation backend used by
[`value_gradient`](@ref), [`value_jacobian`](@ref), `sparse_jacobian` and the
solver-facing derivative evaluators. The package providing the backend (e.g.
Enzyme.jl for `AutoEnzyme()`) must be loaded by the caller.
"""
set_ad_backend!(backend::AbstractADType) = (DEFAULT_AD_BACKEND[] = backend)

"""
    get_ad_backend()

Returns the package-wide default automatic differentiation backend, see
[`set_ad_backend!`](@ref).
"""
get_ad_backend() = DEFAULT_AD_BACKEND[]

import JuMP, MathOptInterface
const MOI = MathOptInterface
using SolverCore: log_header, log_row
using JuMP:
    VariableRef,
    is_binary,
    is_integer,
    has_lower_bound,
    has_upper_bound,
    lower_bound,
    upper_bound,
    start_value,
    ConstraintRef,
    constraint_object,
    AffExpr,
    QuadExpr,
    objective_function,
    objective_sense
using DifferentiableFlatten: flatten, maybeflatten, Unflatten

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

include("models/model.jl")
include("models/vec_model.jl")
include("models/dict_model.jl")
include("models/model_docs.jl")
include("models/jump.jl")
include("models/moi.jl")

# Utilities that depend on the model layer

include("utilities/capabilities.jl")

end
