using NonconvexCore, LinearAlgebra, Test, Zygote

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

m = Model(f)
addvar!(m, [0.0, 0.0], [10.0, 10.0])
add_ineq_constraint!(m, x -> g(x, 2, 0))
add_ineq_constraint!(m, x -> g(x, -1, 1))

@test getmin(m) == fill(0, 2)
@test getmax(m) == fill(10, 2)
@test getmin(m, 1) == 0.0
@test getmax(m, 1) == 10.0
@test getmin(m, 2) == 0.0
@test getmax(m, 2) == 10.0

p = [1.234, 2.345]

val0, grad0 = NonconvexCore.value_gradient(NonconvexCore.getobjective(m), p)
@test val0 == f(p)
@test grad0 == Zygote.gradient(f, p)[1]

val1, grad1 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(m, 1), p)
@test val1 == g(p, 2, 0)
@test grad1 == Zygote.gradient(p -> g(p, 2, 0), p)[1]

val2, grad2 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(m, 2), p)
@test val2 == g(p, -1, 1)
@test grad2 == Zygote.gradient(p -> g(p, -1, 1), p)[1]

vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getineqconstraints(m), p)
@test [val1, val2] == vals
@test [grad1 grad2]' == jac

vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getobjectiveconstraints(m), p)
@test [val0, val1, val2] == vals
@test [grad0 grad1 grad2]' == jac

# A stub optimizer that reports the initial point as the solution. Used to
# test that `optimize` unflattens the result for structured variables.
struct StubOptimizer <: NonconvexCore.AbstractOptimizer end
mutable struct StubWorkspace <: NonconvexCore.Workspace
    model::NonconvexCore.VecModel
    optimizer::StubOptimizer
    x0::AbstractVector
end
function NonconvexCore.Workspace(model::NonconvexCore.VecModel, optimizer::StubOptimizer, x0; kwargs...)
    return StubWorkspace(model, optimizer, copy(x0))
end
function NonconvexCore.optimize!(w::StubWorkspace)
    return NonconvexCore.GenericResult(
        w.optimizer,
        w.x0,
        w.x0,
        0.0,
        0,
        false,
        Tolerance(),
        nothing,
        0,
    )
end

@testset "Structured initial_x in result" begin
    m = Model(x -> x.a + sum(x.b))
    addvar!(m, (a = 0.0, b = [0.0, 0.0]), (a = 1.0, b = [1.0, 1.0]))
    x0 = (a = 0.3, b = [0.4, 0.6])
    r = optimize(m, StubOptimizer(), x0)
    @test r.initial_x == x0
    @test r.minimizer == x0
end

# Wrapper solvers (Ipopt, NLopt, ...) return result types without an
# initial_x field; optimize must not try to restore a field that does not
# exist.
struct StubNoFieldOptimizer <: NonconvexCore.AbstractOptimizer end
mutable struct StubNoFieldWorkspace <: NonconvexCore.Workspace
    model::NonconvexCore.VecModel
    optimizer::StubNoFieldOptimizer
    x0::AbstractVector
end
struct StubNoFieldResult <: NonconvexCore.AbstractResult
    minimizer::Any
end
function NonconvexCore.Workspace(model::NonconvexCore.VecModel, optimizer::StubNoFieldOptimizer, x0; kwargs...)
    return StubNoFieldWorkspace(model, optimizer, copy(x0))
end
function NonconvexCore.optimize!(w::StubNoFieldWorkspace)
    return StubNoFieldResult(w.x0)
end

@testset "Result without initial_x field" begin
    m = Model(x -> x.a + sum(x.b))
    addvar!(m, (a = 0.0, b = [0.0, 0.0]), (a = 1.0, b = [1.0, 1.0]))
    x0 = (a = 0.3, b = [0.4, 0.6])
    r = optimize(m, StubNoFieldOptimizer(), x0)
    @test r.minimizer == x0
    @test !hasproperty(r, :initial_x)
end
