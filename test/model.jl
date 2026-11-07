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

@testset "Non-finite initial point" begin
    struct FiniteCheckDummyOptimizer <: NonconvexCore.AbstractOptimizer end
    m = Model(x -> x[1] < 0.0 ? NaN : x[1])
    addvar!(m, [-10.0], [10.0])
    err = try
        optimize(m, FiniteCheckDummyOptimizer(), [-1.0])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("non-finite", sprint(showerror, err))

    m2 = Model(x -> x[1])
    addvar!(m2, [-10.0], [10.0])
    add_ineq_constraint!(m2, x -> x[1] < -5.0 ? NaN : x[1])
    err2 = try
        optimize(m2, FiniteCheckDummyOptimizer(), [-6.0])
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("inequality constraint #1", sprint(showerror, err2))

    # A finite start passes the check (explicit point and model init).
    m3 = Model(x -> x[1])
    addvar!(m3, [-10.0], [10.0])
    NonconvexCore.check_finite_start!(m3, ([0.0],))
    NonconvexCore.check_finite_start!(m3, ())
end
