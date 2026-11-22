@testitem "Type stability" begin
    using NonconvexCore, Test, LinearAlgebra, SparseArrays
    using ADTypes: ADTypes

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
    @test (@inferred NonconvexCore.CustomGradFunction(x -> x[1] + x[2], x -> [1.0, 1.0])(
        x0
    )) isa Float64
    @test (@inferred NonconvexCore.TraceFunction(sum)(x0)) isa Float64
    @test (@inferred NonconvexCore.LazyJacobian(; jvp=v -> v, symmetric=true) * x0) isa
        Vector{Float64}
    @test (@inferred NonconvexCore.diffy(x -> sum(abs2, x), x0)(x0)) isa Float64

    # Sparse bookkeeping utilities.
    @test (@inferred NonconvexCore.nvalues(nothing)) == 0
    @test (@inferred NonconvexCore.nvalues(zeros(2, 2))) == 4
    rows = zeros(Int, 4)
    cols = zeros(Int, 4)
    @test (@inferred NonconvexCore.fill_indices!(rows, cols, [1.0 0.0; 0.0 1.0])) ==
        (rows, cols)
end

@testitem "Sets: type stability" begin
    using NonconvexCore, Test, LinearAlgebra
    using NonconvexCore:
        getdim,
        set_violation,
        components,
        set_kinds,
        variable_set_kinds,
        variable_dim,
        variable_box,
        variable_init,
        setof,
        tosetconstraint

    # Set constructors (including the mixed-type promotion of Interval).
    @test (@inferred LessThan(0.0)) isa LessThan{Float64}
    @test (@inferred GreaterThan(1)) isa GreaterThan{Int}
    @test (@inferred EqualTo([0.0, 0.0])) isa EqualTo{Vector{Float64}}
    @test (@inferred Interval(0.0, 1.0)) isa Interval{Float64}
    @test (@inferred Interval([0.0], [1])) isa Interval{Vector{Float64}}
    @test (@inferred Nonnegatives(2)) isa Nonnegatives
    @test (@inferred Nonpositives(2)) isa Nonpositives
    @test (@inferred Zeros(2)) isa Zeros
    @test (@inferred Reals(2)) isa Reals
    @test (@inferred Simplex(3)) isa Simplex
    @test (@inferred PSDCone(3)) isa PSDCone
    @test (@inferred Zero()) isa Zero
    @test (@inferred SetProduct(LessThan(0.0), PSDCone(2))) isa SetProduct
    @test (@inferred ChanceSet(LessThan(0.0), [nothing]; epsilon=Zero())) isa ChanceSet

    # Dimensions, decomposition and kind detection.
    @test (@inferred getdim(PSDCone(2))) == 4
    @test (@inferred components(SetProduct(LessThan(0.0), PSDCone(2)))) isa
        Tuple{LessThan{Float64},PSDCone}
    @test (@inferred set_kinds(SetProduct(LessThan(0.0), PSDCone(2)))) ==
        [:product, :ineq, :sd]
    @test (@inferred variable_set_kinds(PSDCone(2))) == [:sd]
    @test (@inferred variable_dim(PSDCone(2))) == 4
    @test (@inferred variable_box(PSDCone(2))) isa Tuple{Vector{Float64},Vector{Float64}}
    @test (@inferred variable_init(PSDCone(2))) == vec(Matrix{Float64}(I, 2, 2))
    @test (@inferred has_psd_component(SetProduct(LessThan(0.0), PSDCone(2)))) === true
    @test (@inferred has_psd_component(LessThan(0.0))) === false

    # Violation and the classic-constraint conversions.
    c = FunctionInSet(
        x -> (x[1] - 1.0, x[2] - 2.0), SetProduct(LessThan(0.0), LessThan(0.0))
    )
    @test (@inferred set_violation(c, [0.0, 0.0])) isa Vector{Float64}
    @test (@inferred setof(NonconvexCore.IneqConstraint(x -> x[1], 0.0))) isa LessThan
    @test (@inferred tosetconstraint(NonconvexCore.SDConstraint(x -> [x[1];;], 1))) isa
        FunctionInSet
    @test tosetconstraint(NonconvexCore.SDConstraint(x -> [x[1];;], 1)).set isa PSDCone
end
