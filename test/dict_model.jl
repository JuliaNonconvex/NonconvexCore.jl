using NonconvexCore, LinearAlgebra, Test, Zygote

@testset "OrderedDict of reals" begin
    f(x) = sqrt(x[:b])
    g(x, a, b) = (a*x[:a] + b)^3 - x[:b]

    m = DictModel(f)
    addvar!(m, :a, 0.0, 10.0)
    addvar!(m, :b, 0.0, 10.0)
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    @test getmin(m) == OrderedDict(:a => 0.0, :b => 0.0)
    @test getmax(m) == OrderedDict(:a => 10.0, :b => 10.0)
    @test getmin(m, :a) == 0.0
    @test getmax(m, :a) == 10.0
    @test getmin(m, :b) == 0.0
    @test getmax(m, :b) == 10.0

    p = OrderedDict(:a => 1.234, :b => 2.345)

    vec_model, _, un = NonconvexCore.tovecmodel(m)
    pvec = NonconvexCore.flatten(p)[1]
    val0, grad0 = NonconvexCore.value_gradient(NonconvexCore.getobjective(vec_model), pvec)
    @test val0 == f(un(pvec))
    @test grad0 == Zygote.gradient(x -> f(un(x)), pvec)[1]

    val1, grad1 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(vec_model, 1), pvec)
    @test val1 == g(un(pvec), 2, 0)
    @test grad1 == Zygote.gradient(pvec -> g(un(pvec), 2, 0), pvec)[1]

    val2, grad2 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(vec_model, 2), pvec)
    @test val2 == g(un(pvec), -1, 1)
    @test grad2 == Zygote.gradient(pvec -> g(un(pvec), -1, 1), pvec)[1]

    vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getineqconstraints(vec_model), pvec)
    @test [val1, val2] == vals
    @test [grad1 grad2]' == jac

    vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getobjectiveconstraints(vec_model), pvec)
    @test [val0, val1, val2] == vals
    @test [grad0 grad1 grad2]' == jac
end

struct S
    a
    b
end
@testset "Vector of structs" begin
    f(x) = sqrt(x[1].b)
    g(x, a, b) = (a*x[1].a + b)^3 - x[1].b

    m = Model(f)
    addvar!(m, S(0.0, 0.0), S(10.0, 10.0))
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    p = [S(1.234, 2.345)]

    vec_model, _, un = NonconvexCore.tovecmodel(m)
    pvec, un = NonconvexCore.flatten(p)
    val0, grad0 = NonconvexCore.value_gradient(NonconvexCore.getobjective(vec_model), pvec)
    @test val0 == f(un(pvec))
    @test grad0 == Zygote.gradient(x -> f(un(x)), pvec)[1]

    val1, grad1 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(vec_model, 1), pvec)
    @test val1 == g(un(pvec), 2, 0)
    @test grad1 == Zygote.gradient(pvec -> g(un(pvec), 2, 0), pvec)[1]

    val2, grad2 = NonconvexCore.value_gradient(NonconvexCore.getineqconstraint(vec_model, 2), pvec)
    @test val2 == g(un(pvec), -1, 1)
    @test grad2 == Zygote.gradient(pvec -> g(un(pvec), -1, 1), pvec)[1]

    vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getineqconstraints(vec_model), pvec)
    @test [val1, val2] == vals
    @test [grad1 grad2]' == jac

    vals, jac = NonconvexCore.value_jacobian(NonconvexCore.getobjectiveconstraints(vec_model), pvec)
    @test [val0, val1, val2] == vals
    @test [grad0 grad1 grad2]' == jac
end
