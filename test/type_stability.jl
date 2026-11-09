@testitem "Type stability" begin
    using JET, NonconvexCore, SparseArrays, LinearAlgebra
    import ADTypes

    # Vector models: objectives and constraints with the default backend.
    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    add_ineq_constraint!(m, x -> x[1] - 0.5)
    add_eq_constraint!(m, x -> x[1] + x[2] - 1)
    vm, x0, _ = NonconvexCore.tovecmodel(m)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.value_gradient(vm.objective, x0)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.value_jacobian(vm.ineq_constraints, x0)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.value_hessian(vm.objective, x0, ADTypes.AutoForwardDiff())
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.sparse_gradient(vm.objective, x0)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.sparse_jacobian(vm.ineq_constraints, x0, ADTypes.AutoReverseDiff())
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.sparse_hessian(vm.objective, x0, ADTypes.AutoReverseDiff())

    # Dict models: variable bookkeeping and the flattening machinery.
    dm = DictModel(x -> x[:a]^2 + x[:b])
    addvar!(dm, :a, -1.0, 1.0)
    addvar!(dm, :b, -1.0, 1.0)
    add_ineq_constraint!(dm, x -> x[:a] - 0.5)
    vm2, x1, _ = NonconvexCore.tovecmodel(dm)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.value_gradient(vm2.objective, x1)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.value_jacobian(vm2.ineq_constraints, x1)
    @test_opt target_modules = (NonconvexCore,) vm2.objective(x1)
    @test_opt target_modules = (NonconvexCore,) vm2.ineq_constraints(x1)

    # Function modifiers.
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.CustomGradFunction(x -> x[1] + x[2], x -> [1.0, 1.0])(x0)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.TraceFunction(sum)(x0)
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.LazyJacobian(; jvp = v -> v, symmetric = true) * x0

    # Preparation.
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.prepare_gradient(vm.objective, x0, ADTypes.AutoForwardDiff())
    @test_opt target_modules = (NonconvexCore,) NonconvexCore.prepare_jacobian(vm.ineq_constraints, x0, ADTypes.AutoForwardDiff())
end
