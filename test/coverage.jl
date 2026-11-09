using NonconvexCore, Test, LinearAlgebra, SparseArrays
import ADTypes
import DifferentiationInterface as DI
using Zygote, ForwardDiff, ReverseDiff
import ChainRulesCore
using ChainRulesCore: NoTangent
using ConvergencePlots
import MathOptInterface as MOI

const NC = NonconvexCore

struct StubOpt <: NC.AbstractOptimizer end

struct TestOptions
    tol::NC.Tolerance
    convcriteria
    verbose::Bool
end

mutable struct TestWorkspace <: NC.Workspace
    solution
    model
    options
    x0::Vector{Float64}
    iter::Int
end
NC.solver_capabilities(::StubOpt) = NC.ConstraintCapabilities(
    linear_eq = true, nonlinear_eq = true,
    linear_ineq = true, nonlinear_ineq = true,
    linear_sd = true, nonlinear_sd = true,
    box = true, continuous = true, binary = true, integer = true,
    infinite_bounds = true,
)

# ------------------------------------------------------------------
# Sparse Jacobian/Hessian bookkeeping utilities (src/models/moi.jl)
# ------------------------------------------------------------------
@testset "sparse bookkeeping utilities" begin
    @test NC.nvalues(nothing) == 0
    @test NC.nvalues(zeros(2, 3)) == 6
    @test NC.nvalues(LowerTriangular(zeros(3, 3))) == 6
    @test NC.nvalues(sparse([1.0 0.0; 2.0 3.0])) == 3
    @test NC.nvalues(LowerTriangular(sparse([1.0 0.0; 2.0 3.0]))) == 3

    # fill_indices! for dense matrices (column-major structure)
    rows = zeros(Int, 6)
    cols = zeros(Int, 6)
    NC.fill_indices!(rows, cols, [1.0 0.0; 0.0 1.0])
    @test rows[1:4] == [1, 2, 1, 2]
    @test cols[1:4] == [1, 1, 2, 2]

    # fill_indices! for lower-triangular dense
    rows .= 0
    cols .= 0
    NC.fill_indices!(rows, cols, LowerTriangular([1.0 0.0; 2.0 3.0]))
    @test sort(collect(zip(rows[1:3], cols[1:3]))) == [(1, 1), (2, 1), (2, 2)]

    # fill_indices! for sparse CSC
    rows .= 0
    cols .= 0
    NC.fill_indices!(rows, cols, sparse([1.0 0.0; 2.0 3.0]))
    @test sort(collect(zip(rows[1:3], cols[1:3]))) == [(1, 1), (2, 1), (2, 2)]

    # fill_indices! for lower-triangular sparse
    rows .= 0
    cols .= 0
    NC.fill_indices!(rows, cols, LowerTriangular(sparse([1.0 0.0; 2.0 3.0])))
    @test sort(collect(zip(rows[1:3], cols[1:3]))) == [(1, 1), (2, 1), (2, 2)]

    # offsets
    rows2 = zeros(Int, 6)
    cols2 = zeros(Int, 6)
    NC.fill_indices!(rows2, cols2, [1.0 0.0; 0.0 1.0], offset = 1, row_offset = 5)
    @test rows2[2] == 6 && cols2[2] == 1

    # add_values!
    v = zeros(4)
    NC.add_values!(v, [1.0 2.0; 3.0 4.0])
    @test v == [1.0, 3.0, 2.0, 4.0]
    v = zeros(3)
    NC.add_values!(v, LowerTriangular([1.0 0.0; 2.0 3.0]))
    @test v == [1.0, 2.0, 3.0]
    v = zeros(3)
    NC.add_values!(v, sparse([1.0 0.0; 2.0 3.0]))
    @test v == [1.0, 2.0, 3.0]
    v = zeros(3)
    NC.add_values!(v, LowerTriangular(sparse([1.0 0.0; 2.0 3.0])))
    @test v == [1.0, 2.0, 3.0]
    v = zeros(4)
    NC.add_values!(v, sparse([1.0 0.0; 2.0 3.0]), factor = 2.0, offset = 1)
    @test v == [0.0, 2.0, 4.0, 6.0]

    # _dot
    @test NC._dot(x -> [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]) == 11.0
    @test NC._dot(nothing, [0.0], [1.0]) == 0.0
    m = Model()
    addvar!(m, [-1.0], [1.0])
    add_ineq_constraint!(m, x -> -x[1])
    add_ineq_constraint!(m, x -> 2x[1] - 1)
    vm, xv, _ = NC.tovecmodel(m)
    s = NC._dot(vm.ineq_constraints, xv, [1.0, 2.0])
    @test s ≈ -xv[1] + 2 * (2xv[1] - 1)
end

# ------------------------------------------------------------------
# Callbacks (src/utilities/callbacks.jl)
# ------------------------------------------------------------------
@testset "callbacks" begin
    @test NC.NoCallback()(nothing) === nothing
    cb = NC.PlottingCallback(5)
    @test cb.iter == 0
    cb(1)  # non-Solution argument: the plotting-free fallback method
    lazy = NC.LazyPlottingCallback(10; show_plot = false, save_plot = "x.png")
    @test lazy.iter == 0
end

# ------------------------------------------------------------------
# Explicit preparation and diffy (src/functions/prepare.jl)
# ------------------------------------------------------------------
@testset "explicit preparation" begin
    g2 = z -> [z[1] + z[2], z[3]^2]
    x2 = [1.0, 2.0, 3.0]
    # prepare_hessian
    NC.prepare_hessian(z -> sum(abs2, z), x2, ADTypes.AutoForwardDiff())
    H = NC.value_hessian(z -> sum(abs2, z), x2, ADTypes.AutoForwardDiff())
    @test H ≈ 2I
    # prepare_sparse_*
    NC.prepare_sparse_jacobian(g2, x2, ADTypes.AutoReverseDiff())
    J = NC.sparse_jacobian(g2, x2, ADTypes.AutoReverseDiff())
    @test Matrix(J) ≈ [1.0 1.0 0.0; 0.0 0.0 6.0]
    NC.prepare_sparse_hessian(z -> sum(abs2, z), x2, ADTypes.AutoReverseDiff())
    Hs = NC.sparse_hessian(z -> sum(abs2, z), x2, ADTypes.AutoReverseDiff())
    @test Matrix(Hs) ≈ 2I
    # diffy scalar: value, gradient through the prepared backend, rrule
    f2 = z -> sum(abs2, z)
    F = NC.diffy(f2, x2)
    @test F(x2) == f2(x2)
    @test Zygote.gradient(F, x2)[1] ≈ 2x2
    # diffy vector rrule
    G = NC.diffy(g2, x2, ADTypes.AutoForwardDiff())
    @test Zygote.jacobian(G, x2)[1] ≈ [1.0 1.0 0.0; 0.0 0.0 6.0]
    # prepare_model with options
    m = Model(z -> sum(abs2, z))
    addvar!(m, -ones(3), ones(3))
    add_ineq_constraint!(m, z -> z[1] + z[2])
    vm = NC.prepare_model(m, ones(3))
    @test NC.value_gradient(NC.getobjective(vm), ones(3))[2] ≈ 2ones(3)
    vm = NC.prepare_model(m, ones(3); hessian = false, jacobian = false, gradient = false)
    @test vm isa NC.VecModel
end

# ------------------------------------------------------------------
# Capabilities (src/utilities/capabilities.jl)
# ------------------------------------------------------------------
@testset "capabilities helpers" begin
    base = NC.ConstraintCapabilities(linear_eq = true, nonlinear_ineq = false)
    merged = NC.merge_capabilities(base; nonlinear_ineq = true, integer = true)
    @test merged.linear_eq === true
    @test merged.nonlinear_ineq === true
    @test merged.integer === true
    @test merged.linear_sd === missing
    m2 = NC.merge_capabilities(nothing)
    @test m2.linear_eq === missing
    @test occursin("supported", NC._capability_string(:linear_eq, true))
    @test occursin("unsupported", NC._capability_string(:linear_eq, false))
    @test occursin("not declared", NC._capability_string(:linear_eq, missing))
end

# ------------------------------------------------------------------
# Model accessors and mutation (src/models/model.jl, functions.jl)
# ------------------------------------------------------------------
@testset "model accessors" begin
    m = Model(x -> x[1]^2)
    addvar!(m, [-1.0, -1.0], [1.0, 1.0], integer = [true, false])
    add_ineq_constraint!(m, x -> -x[1])
    add_eq_constraint!(m, x -> x[2])
    add_sd_constraint!(m, x -> [x[1] 0.0; 0.0 x[2]])
    @test NC.getdim(NC.getobjective(m)) == 1
    @test NC.getnineqconstraints(m) == 1
    @test NC.getneqconstraints(m) == 1
    @test NC.getnsdconstraints(m) == 2
    @test NC.getnconstraints(m) == 4
    @test NC.getnvars(m) == 2
    @test NC.getdim(m) == (4, 2)
    @test NC.isinteger(m, 1)
    @test !NC.isinteger(m, 2)
    @test NC.geteqconstraint(m, 1) == NC.geteqconstraints(m).fs[1]
    @test NC.getineqconstraint(m, 1) == NC.getineqconstraints(m).fs[1]
    NC.setmin!(m, [-2.0, -2.0])
    @test NC.getmin(m) == [-2.0, -2.0]
    NC.setmin!(m, 1, -3.0)
    @test NC.getmin(m)[1] == -3.0
    NC.setmax!(m, [2.0, 2.0])
    @test NC.getmax(m) == [2.0, 2.0]
    NC.setmax!(m, 2, 3.0)
    @test NC.getmax(m)[2] == 3.0
    NC.setbox!(m, [-4.0, -4.0], [4.0, 4.0])
    @test NC.getmin(m) == [-4.0, -4.0] && NC.getmax(m) == [4.0, 4.0]
    NC.setbox!(m, 1, -5.0, 5.0)
    @test NC.getmin(m)[1] == -5.0 && NC.getmax(m)[1] == 5.0
    m2 = NC.set_objective(m, x -> sum(abs2, x))
    @test m2.objective !== m.objective
    # vector add_*_constraint!
    NC.add_ineq_constraint!(m, NC.IneqConstraint[NC.IneqConstraint(x -> -x[2], 0.0)])
    NC.add_eq_constraint!(m, NC.EqConstraint[NC.EqConstraint(x -> x[1], 0.0)])
    @test NC.getnineqconstraints(m) == 2
    @test NC.getneqconstraints(m) == 2
end

# ------------------------------------------------------------------
# VecModel paths (src/models/vec_model.jl)
# ------------------------------------------------------------------
@testset "vec model paths" begin
    m = Model(x -> x[1]^2)
    addvar!(m, [-1.0], [1.0])
    add_ineq_constraint!(m, x -> -x[1])
    vm, xv, un = NC.tovecmodel(m)
    @test NC.isfeasible(vm, [0.0])
    @test !NC.isfeasible(vm, [2.0])
    dm = NC.DictModel(x -> x[:a])
    NC.addvar!(dm, :a, -1.0, 1.0)
    vm2, xv2, _ = NC.tovecmodel(dm)
    flat = NC.getobjective(vm2).f
    @test NC.getinputunflatten(flat) isa Function
    @test NC.getrawfunction(flat) isa Function
    @test flat(xv2) ≈ xv2[1]
    # tovecfunc paths
    f = x -> x[1] + 1
    flat_f, vx, unf = NC.tovecfunc(f, xv)
    @test unf(flat_f(xv)) ≈ f(xv)
    g = (x, a) -> a * x[1]
    flat_g, vx2, unf2 = NC.tovecfunc(g, xv, 2.0)
    @test unf2(flat_g(vx2)) ≈ g(xv, 2.0)
    # default init for infinite bounds (applied by getinit on the VecModel)
    m2 = Model(x -> x[1])
    addvar!(m2, -Inf, Inf)
    vm2, _, _ = NC.tovecmodel(m2)
    @test NC.getinit(vm2) == [0.0]
    m3 = Model(x -> x[1])
    addvar!(m3, -Inf, 1.0)
    vm3, _, _ = NC.tovecmodel(m3)
    @test NC.getinit(vm3) == [0.0]
    m4 = Model(x -> x[1])
    addvar!(m4, 1.0, Inf, init = Inf)
    vm4, _, _ = NC.tovecmodel(m4)
    @test NC.getinit(vm4) == [2.0]
end

# ------------------------------------------------------------------
# Common utilities (src/common.jl)
# ------------------------------------------------------------------
@testset "common utilities" begin
    m = Model(x -> x[1]^2)
    addvar!(m, [-1.0], [1.0])
    vm, xv, _ = NC.tovecmodel(m)
    stub = StubOpt()
    w = TestWorkspace(nothing, vm, TestOptions(NC.Tolerance(), NC.KKTCriteria(), false), copy(xv), 0)
    NC.reset!(w, xv)
    @test w.x0 == xv
    NC.reset!(w, nothing)
    @test w.x0 == xv
    @test_throws String NC.optimize!(w)
    cs = NC.ConvergenceState{Float64}()
    @test cs isa NC.ConvergenceState
    x = [0.5, 0.5]
    NC.clamp_and_evaluate!(vm, x)
    @test all(-1.0 .<= x .<= 1.0)
end

# ------------------------------------------------------------------
# Convergence criteria (src/utilities/convergence.jl)
# ------------------------------------------------------------------
@testset "convergence criteria" begin
    m = Model(x -> x[1]^2)
    addvar!(m, [-1.0], [1.0])
    add_ineq_constraint!(m, x -> -x[1])
    vm, xv, _ = NC.tovecmodel(m)
    stub = StubOpt()
    for criteria in (NC.GenericCriteria(), NC.KKTCriteria(), NC.ScaledKKTCriteria(), NC.IpoptCriteria())
        convstate = NC.ConvergenceState{Float64}()
        sol = NC.Solution([0.0], [0.5], [0.0], 1.0, 0.25, [0.0], [0.0], reshape([-1.0], 1, 1), convstate)
        w = TestWorkspace(sol, vm, TestOptions(NC.Tolerance(), criteria, true), copy(xv), 0)
        NC.assess_convergence!(w)
        NC.hasconverged(w.solution)
        NC.correctsolution!(w.solution, w.model, w.options)
    end
end

# ------------------------------------------------------------------
# Function modifiers (src/utilities/custom.jl, lazy.jl, trace.jl)
# ------------------------------------------------------------------
@testset "function modifiers" begin
    # CustomGradFunction with a LazyJacobian gradient
    lazy_jac = NC.LazyJacobian(; jvp = v -> [v[1] - v[2]], jtvp = w -> [w[1], -w[1]])
    cf = NC.CustomGradFunction(x -> [x[1] - x[2]], x -> lazy_jac)
    @test Zygote.jacobian(cf, [1.0, 2.0])[1] ≈ [1.0 -1.0]
    # CustomHessianFunction: full Hessian and hvp
    hfun = x -> [2.0 0.0; 0.0 2.0]
    ch = NC.CustomHessianFunction(x -> sum(abs2, x), x -> 2x, hfun)
    @test Zygote.hessian(ch, [1.0, 2.0]) ≈ hfun([1.0, 2.0])
    chvp = NC.CustomHessianFunction(x -> sum(abs2, x), x -> 2x, (x, v) -> 2v; hvp = true)
    @test Zygote.gradient(z -> sum(chvp(z)), [1.0, 2.0])[1] ≈ 2 .* [1.0, 2.0]
    # LazyJacobian operators
    j = NC.LazyJacobian(; jvp = v -> [v[1] + v[2]], jtvp = w -> [w[1], w[1]])
    @test transpose(j) * [1.0] == j' * [1.0]
    @test j * [1.0, 2.0] == [3.0]
    @test [1.0] * j == j.jtvp([1.0])'
    jsym = NC.LazyJacobian(; jvp = v -> v, symmetric = true)
    @test jsym' === jsym
    @test transpose(jsym) === jsym
    @test jsym * [1.0, 2.0] == [1.0, 2.0]
    # TraceFunction gradient tracing
    tr = NC.TraceFunction(sum; on_call = true, on_grad = true)
    @test tr.on_grad == true
    tr([2.0])
    Zygote.gradient(tr, [2.0])
end

# ------------------------------------------------------------------
# @params macro (src/utilities/params.jl)
# ------------------------------------------------------------------
@testset "@params" begin
    NC.@params struct ParamStruct
        a::Int
        b::Float64
    end
    p = ParamStruct(1, 1.0)
    @test p.a == 1
    @test p.b == 1.0
end

# ------------------------------------------------------------------
# Sparse gradient variants (src/functions/sparse.jl)
# ------------------------------------------------------------------
@testset "sparse gradient variants" begin
    x = [1.0, 2.0, 3.0]
    g = NC.sparse_gradient(z -> sum(abs2, z), x)
    @test g ≈ sparsevec([2.0, 4.0, 6.0])
    # scalar, matrix and adjoint outputs through _sparsevec
    @test NC._sparsevec(1.0) == [1.0]
    @test NC._sparsevec([1.0, 2.0]) == [1.0, 2.0]
    @test NC._sparsevec([1.0 2.0; 3.0 4.0]) == [1.0, 3.0, 2.0, 4.0]
    @test NC._sparsevec(sparsevec([1.0, 0.0, 2.0])) == [1.0, 0.0, 2.0]
    @test NC._sparsevec([1.0 2.0]') == [1.0, 2.0]
end

# ------------------------------------------------------------------
# Sparse gradient variants (src/functions/sparse.jl)
# ------------------------------------------------------------------
@testset "sparse gradient variants" begin
    x = [1.0, 2.0, 3.0]
    g = NC.sparse_gradient(z -> sum(abs2, z), x)
    @test g ≈ sparsevec([2.0, 4.0, 6.0])
    # scalar, matrix and adjoint outputs through _sparsevec
    @test NC._sparsevec(1.0) == [1.0]
    @test NC._sparsevec([1.0, 2.0]) == [1.0, 2.0]
    @test NC._sparsevec([1.0 2.0; 3.0 4.0]) == [1.0, 3.0, 2.0, 4.0]
    @test NC._sparsevec(sparsevec([1.0, 0.0, 2.0])) == [1.0, 0.0, 2.0]
    @test NC._sparsevec([1.0 2.0]') == [1.0, 2.0]
end

# ------------------------------------------------------------------
# Remaining live paths
# ------------------------------------------------------------------
@testset "remaining live paths" begin
    # ConvergenceState default constructor
    cs = NC.ConvergenceState{Float64}()
    @test cs isa NC.ConvergenceState

    # getdim of Objective, empty VectorOfFunctions tuple construction
    m = Model(x -> x[1])
    addvar!(m, [-1.0], [1.0])
    @test NC.getdim(NC.getobjective(m)) == 1
    @test NC.VectorOfFunctions(()) isa NC.VectorOfFunctions

    # prepare_model with an equality constraint (prepares the eq jacobian)
    m2 = Model(z -> sum(abs2, z))
    addvar!(m2, -ones(2), ones(2))
    add_eq_constraint!(m2, z -> z[1] + z[2])
    vm2 = NC.prepare_model(m2, ones(2))
    @test NC.value_jacobian(NC.geteqconstraints(vm2), ones(2))[2] ≈ reshape([1.0, 1.0], 1, 2)

    # _sparse_reshape sparse-vector branch
    @test Matrix(NC._sparse_reshape(sparsevec([1.0, 0.0, 2.0]), 3, 1))[:] == [1.0, 0.0, 2.0]
    @test NC._sparse_reshape(sparsevec(Float64[]), 2, 2) == zeros(2, 2)

    # VecModel addvar! single and vector forms
    vm0, xv0, _ = NC.tovecmodel(m2)
    NC.addvar!(vm0, -1.0, 1.0, init = 0.0, integer = true)
    NC.addvar!(vm0, [-2.0, -2.0], [2.0, 2.0], init = [0.0, 0.0], integer = [false, false])
    @test NC.getmin(vm0)[end-2:end] == [-1.0, -2.0, -2.0]
    @test NC.getmax(vm0)[end-2:end] == [1.0, 2.0, 2.0]
    @test vm0.integer[end-2:end] == [true, false, false]

    # CustomGradFunction with g === nothing (zero gradients)
    cg = NC.CustomGradFunction(x -> x[1] + 2x[2], nothing)
    @test Zygote.gradient(cg, [1.0, 2.0])[1] == [0.0, 0.0]
    cgv = NC.CustomGradFunction(x -> [x[1], x[2]], nothing)
    @test Zygote.jacobian(cgv, [1.0, 2.0])[1] == zeros(2, 2)
    # frule paths
    cf = NC.CustomGradFunction(x -> x[1] + x[2], x -> [1.0, 1.0])
    @test ForwardDiff.derivative(t -> cf([t, 0.0]), 1.0) ≈ 1.0
    ch = NC.CustomHessianFunction(x -> sum(abs2, x), x -> 2x, x -> [2.0 0.0; 0.0 2.0])
    @test ForwardDiff.derivative(t -> ch([t, 0.0]), 1.0) ≈ 2.0

    # LazyJacobian error + symmetric-from-jtvp + transposed operators
    @test_throws ArgumentError NC.LazyJacobian()
    lsym = NC.LazyJacobian(; jtvp = w -> w, symmetric = true)
    @test lsym.jvp === lsym.jtvp
    j = NC.LazyJacobian(; jvp = v -> v, jtvp = w -> w)
    jt = transpose(j)
    @test transpose(jt) === j
    @test jt * [1.0, 2.0] == [1.0, 2.0]
    @test ([1.0, 2.0] * jt) == j.jtvp([1.0, 2.0])

    # TraceFunction frule (with a function that has a ChainRules frule)
    tr = NC.TraceFunction(exp; on_call = true, on_grad = true)
    @test ForwardDiff.derivative(tr, 2.0) ≈ exp(2.0)

    # @params with a parametrized subtype header
    NC.@params struct ParamSub{T} <: NC.AbstractOptimizer
        a::T
    end
    ps = ParamSub(1.0)
    @test ps.a == 1.0
    @test ps isa NC.AbstractOptimizer

    # _dot with nothing
    @test NC._dot(nothing, [1.0], [2.0]) == 0.0
    # nvalues of nothing
    @test NC.nvalues(nothing) == 0
end
