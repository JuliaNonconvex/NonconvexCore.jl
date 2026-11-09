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
    get_ad_backend,
    prepare_gradient,
    prepare_jacobian,
    prepare_hessian,
    prepare_sparse_jacobian,
    prepare_sparse_hessian,
    prepare_model,
    diffy,
    JuMPSymbolicFunction,
    TraceFunction,
    CustomGradFunction,
    CustomHessianFunction,
    LazyJacobian,
    ImplicitFunction,
    symbolify,
    jump_symbolic_gradient,
    jump_symbolic_hessian,
    jump_symbolic_function

using Parameters, Zygote, ReverseDiff, ChainRulesCore, ForwardDiff
using LinearMaps: LinearMap
using IterativeSolvers: gmres
using SparseArrays, Reexport, Requires
@reexport using LinearAlgebra, OrderedCollections
@reexport using ADTypes
import DifferentiationInterface
const DI = DifferentiationInterface
import MacroTools
using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity, hessian_sparsity
using SparseMatrixColorings: GreedyColoringAlgorithm
using Reexport, Setfield

"""
    DEFAULT_AD_BACKEND

The automatic differentiation backend used by default wherever NonconvexCore
computes derivatives, e.g. in `value_gradient`, `value_jacobian`,
`sparse_jacobian` and the solver-facing derivative evaluators. It is initially
`ADTypes.AutoZygote()`, whose ChainRules-based design supports arbitrary
argument types (dictionaries, structs, arrays) through the model-flattening
machinery. Change it with [`set_ad_backend!`](@ref), or pass a backend
explicitly to the functions above.
"""
const DEFAULT_AD_BACKEND = Ref{AbstractADType}(AutoZygote())

"""
    set_ad_backend!(backend::ADTypes.AbstractADType)

Sets the package-wide default automatic differentiation backend used by
[`value_gradient`](@ref NonconvexCore.value_gradient), [`value_jacobian`](@ref NonconvexCore.value_jacobian), `sparse_jacobian` and the
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

import MathOptInterface
const MOI = MathOptInterface
using SolverCore: log_header, log_row
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
include("models/jumpsymbolic.jl")
include("models/moi.jl")
include("functions/prepare.jl")

# Utilities that depend on the model layer

include("utilities/capabilities.jl")
include("utilities/forwarddiff_frule.jl")
include("utilities/lazy.jl")
include("utilities/trace.jl")
include("utilities/custom.jl")
include("utilities/implicit.jl")

end
