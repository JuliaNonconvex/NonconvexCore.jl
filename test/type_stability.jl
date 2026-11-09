@testitem "Type stability" begin
    using NonconvexCore, Test, LinearAlgebra, SparseArrays
    import ADTypes

    # NOTE: the return types of the DifferentiationInterface-backed
    # value_*/sparse_* functions are limited by DifferentiationInterface's
    # own inference (its lazy preparation is data-dependent), so this item
    # checks type stability of the concrete NonconvexCore machinery instead:
    # model construction, flattening, evaluation and the function modifiers.

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(m, x -> x[1] - 0.5)
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    vm, x0, _ = NonconvexCore.tovecmodel(m)

    @test (@inferred NonconvexCore.getmin(vm)) isa Vector{Float64}
    @test (@inferred NonconvexCore.getmax(vm)) isa Vector{Float64}
    @test (@inferred NonconvexCore.getinit(vm)) isa Vector{Float64}
    @test (@inferred vm.objective(x0)) isa Float64
    @test (@inferred vm.ineq_constraints(x0)) isa Vector{Float64}
    @test (@inferred vm.eq_constraints(x0)) isa Vector{Float64}

    # Dict models: variable bookkeeping and the flattening machinery.
    dm = DictModel(x -> x[:a]^2 + x[:b])
    addvar!(dm, :a, -1.0, 1.0)
    addvar!(dm, :b, -1.0, 1.0)
    add_ineq_constraint!(dm, x -> x[:a] - 0.5)
    vm2, x1, _ = NonconvexCore.tovecmodel(dm)
    @test (@inferred vm2.objective(x1)) isa Float64
    @test (@inferred vm2.ineq_constraints(x1)) isa Vector{Float64}
    flat = NonconvexCore.getobjective(vm2).f
    @test (@inferred NonconvexCore.getinputunflatten(flat)) isa Function
    @test (@inferred NonconvexCore.getrawfunction(flat)) isa Function

    # Function modifiers.
    @test (@inferred NonconvexCore.CustomGradFunction(x -> x[1] + x[2], x -> [1.0, 1.0])(x0)) isa Float64
    @test (@inferred NonconvexCore.TraceFunction(sum)(x0)) isa Float64
    @test (@inferred NonconvexCore.LazyJacobian(; jvp = v -> v, symmetric = true) * x0) isa Vector{Float64}
    @test (@inferred NonconvexCore.diffy(x -> sum(abs2, x), x0)(x0)) isa Float64

    # Sparse bookkeeping utilities.
    @test (@inferred NonconvexCore.nvalues(nothing)) == 0
    @test (@inferred NonconvexCore.nvalues(zeros(2, 2))) == 4
    rows = zeros(Int, 4)
    cols = zeros(Int, 4)
    @test (@inferred NonconvexCore.fill_indices!(rows, cols, [1.0 0.0; 0.0 1.0])) == (rows, cols)
end
