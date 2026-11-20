@testitem "JuMP-converted functions reject ForwardDiff.Dual inputs" begin
    using Test, NonconvexCore
    using JuMP: JuMP
    using ForwardDiff: ForwardDiff

    model = JuMP.Model()
    JuMP.@variable(model, x)
    # sin(x) forces the JuMP-converted (evaluator) path; a pure quadratic
    # objective goes through the closed-form channel instead
    JuMP.@objective(model, Min, x^2 + sin(x))
    JuMP.@constraint(model, x >= 1)
    dm = DictModel(model)
    vm, xv, _ = NonconvexCore.tovecmodel(dm)
    obj = NonconvexCore.getobjective(vm)
    ineq = vm.ineq_constraints.fs[1]

    # Float64 evaluation works
    @test obj(xv) isa Float64
    @test ineq(xv)[1] isa Float64

    # Nested AD (ForwardDiff-over-Zygote) must fail loudly instead of
    # silently corrupting the evaluator's hard-Float64 storage
    dx = ForwardDiff.Dual.(xv, zeros(length(xv)))
    @test_throws ArgumentError obj(dx)
    @test_throws ArgumentError ineq(dx)
end
