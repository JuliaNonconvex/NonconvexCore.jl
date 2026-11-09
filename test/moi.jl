@testitem "MOI evaluator" begin
using NonconvexCore, LinearAlgebra, Test
import JuMP, MathOptInterface, Zygote
const MOI = MathOptInterface

f(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
g(x) = x[1] + x[2] - 1.0
h(x) = 2x[1] - x[2]

function make_problem(; first_order, integer = false)
    m = Model(f)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m, g)
    add_eq_constraint!(m, h)
    integer && setinteger!(m, 1, true)
    vm, x0, _ = NonconvexCore.tovecmodel(m)
    optimizer = () -> MOI.Utilities.MockOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    )
    problem, _ = NonconvexCore.get_jump_problem(vm, x0; optimizer, first_order)
    return problem, x0
end

@testset "MOI evaluator - second order" begin
    problem, x0 = make_problem(; first_order = false)
    evaluator = problem.evaluator

    @test MOI.features_available(evaluator) == [:Grad, :Jac, :Hess, :JacVec, :HessVec]
    @test MOI.initialize(evaluator, [:Grad, :Jac, :Hess, :JacVec, :HessVec]) === nothing
    @test_throws ErrorException MOI.initialize(evaluator, [:Bad])

    @test MOI.eval_objective(evaluator, x0) == f(x0)
    grad = zeros(2)
    MOI.eval_objective_gradient(evaluator, grad, x0)
    @test grad ≈ Zygote.gradient(f, x0)[1]

    gvals = zeros(2)
    MOI.eval_constraint(evaluator, gvals, x0)
    @test gvals ≈ [g(x0), h(x0)]

    structure = MOI.jacobian_structure(evaluator)
    @test structure == [(1, 1), (1, 2), (2, 1), (2, 2)]
    J = zeros(4)
    MOI.eval_constraint_jacobian(evaluator, J, x0)
    @test J ≈ [1.0, 1.0, 2.0, -1.0]

    hess_structure = MOI.hessian_lagrangian_structure(evaluator)
    @test hess_structure == [(1, 1), (2, 2)]
    H = zeros(2)
    MOI.eval_hessian_lagrangian(evaluator, H, x0, 1.5, [2.0, 3.0])
    @test H ≈ [3.0, 3.0]

    # Jac-vec and Hess-vec products
    w = [0.3, -0.7]
    y = zeros(2)
    MOI.eval_constraint_jacobian_product(evaluator, y, x0, w)
    @test y ≈ [1.0 1.0; 2.0 -1.0] * w
    z = zeros(2)
    MOI.eval_constraint_jacobian_transpose_product(evaluator, z, x0, w)
    @test z ≈ transpose([1.0 1.0; 2.0 -1.0]) * w
    v = [0.2, 0.4]
    hv = zeros(2)
    MOI.eval_hessian_lagrangian_product(evaluator, hv, x0, v, 1.5, [2.0, 3.0])
    @test hv ≈ [3.0 0.0; 0.0 3.0] * v

    @test length(problem.vars) == 2
    @test MOI.get(JuMP.backend(problem.model), MOI.NumberOfVariables()) == 2
end

@testset "MOI evaluator - first order" begin
    problem, x0 = make_problem(; first_order = true)
    evaluator = problem.evaluator
    @test MOI.initialize(evaluator, [:Grad, :Jac]) === nothing
    @test isempty(MOI.hessian_lagrangian_structure(evaluator))
end

@testset "MOI evaluator - integer variables" begin
    problem, x0 = make_problem(; first_order = true, integer = true)
    model = JuMP.backend(problem.model)
    @test MOI.get(model, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Integer}()) == 1
end
end
