@testitem "JuMP" begin
    using Test, NonconvexCore
    using JuMP: JuMP, MOI

    @testset "Symbol variable names" begin
        model = JuMP.Model()
        JuMP.@variable(model, x >= 0)
        JuMP.@variable(model, 0 <= y <= 3)
        JuMP.@objective(model, Min, 20y)
        JuMP.@constraint(model, c1, 6x + 8y >= 100)
        JuMP.@constraint(model, c2, 7x + 12y >= 120)
        JuMP.@constraint(model, c3, x + y == 25)
        JuMP.@constraint(model, c4, x + y <= 50)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x = NonconvexCore.getinit(dict_model)

        @test getmin(dict_model) == OrderedDict(:x => 0.0, :y => 0.0)
        @test getmax(dict_model) == OrderedDict(:x => Inf, :y => 3.0)
        @test obj(x) == 20 * x[:y]
        @test ineq(x) == [
            100 - 6 * x[:x] - 8 * x[:y], 120 - 7 * x[:x] - 12 * x[:y], x[:x] + x[:y] - 50
        ]
        @test eq(x) == [x[:x] + x[:y] - 25]

        vec_model, _, _ = NonconvexCore.tovecmodel(dict_model)
        vec_x = NonconvexCore.getinit(vec_model)
        vec_obj = NonconvexCore.getobjective(vec_model)
        vec_ineq = NonconvexCore.getineqconstraints(vec_model)
        vec_eq = NonconvexCore.geteqconstraints(vec_model)

        val0, grad0 = NonconvexCore.value_gradient(vec_obj, vec_x)
        val1, jac1 = NonconvexCore.value_jacobian(vec_ineq, vec_x)
        val2, jac2 = NonconvexCore.value_jacobian(vec_eq, vec_x)

        @test val0 == obj(x)
        @test val1 == ineq(x)
        @test val2 == eq(x)

        @test grad0 == [0, 20]
        @test jac1 == [-6 -8; -7 -12; 1 1]
        @test jac2 == [1 1;]
    end

    @testset "x[ind] variable names 1" begin
        model = JuMP.Model()
        lbs = [0.0, 0.0]
        ubs = [Inf, 3.0]
        JuMP.@variable(model, x[i = 1:2], lower_bound=lbs[i], upper_bound=ubs[i])
        JuMP.@objective(model, Min, 12x[1])
        JuMP.@constraint(model, c1, 6x[1] + 8x[2] >= 100)
        JuMP.@constraint(model, c2, 7x[1] + 12x[2] >= 120)
        JuMP.@constraint(model, c3, x[1] + x[2] == 25)
        JuMP.@constraint(model, c4, x[1] + x[2] <= 50)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x = NonconvexCore.getinit(dict_model)

        @test getmin(dict_model) == OrderedDict(:x => [0.0, 0.0])
        @test getmax(dict_model) == OrderedDict(:x => [Inf, 3.0])
        @test obj(x) == 12 * x[:x][1]
        @test ineq(x) == [
            100 - 6 * x[:x][1] - 8 * x[:x][2],
            120 - 7 * x[:x][1] - 12 * x[:x][2],
            x[:x][1] + x[:x][2] - 50,
        ]
        @test eq(x) == [x[:x][1] + x[:x][2] - 25]

        vec_model, _, _ = NonconvexCore.tovecmodel(dict_model)
        vec_x = NonconvexCore.getinit(vec_model)
        vec_obj = NonconvexCore.getobjective(vec_model)
        vec_ineq = NonconvexCore.getineqconstraints(vec_model)
        vec_eq = NonconvexCore.geteqconstraints(vec_model)

        val0, grad0 = NonconvexCore.value_gradient(vec_obj, vec_x)
        val1, jac1 = NonconvexCore.value_jacobian(vec_ineq, vec_x)
        val2, jac2 = NonconvexCore.value_jacobian(vec_eq, vec_x)

        @test val0 == obj(x)
        @test val1 == ineq(x)
        @test val2 == eq(x)

        @test grad0 == [12, 0]
        @test jac1 == [-6 -8; -7 -12; 1 1]
        @test jac2 == [1 1;]
    end

    @testset "x[ind] variable names 2" begin
        model = JuMP.Model()
        lbs = [0.0, 0.0]
        ubs = [Inf, 3.0]
        JuMP.@variable(model, x[i = 2:-1:1], lower_bound=lbs[i], upper_bound=ubs[i])
        JuMP.@objective(model, Min, 12x[1])
        JuMP.@constraint(model, c1, 6x[1] + 8x[2] >= 100)
        JuMP.@constraint(model, c2, 7x[1] + 12x[2] >= 120)
        JuMP.@constraint(model, c3, x[1] + x[2] == 25)
        JuMP.@constraint(model, c4, x[1] + x[2] <= 50)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x = NonconvexCore.getinit(dict_model)

        @test getmin(dict_model) ==
            OrderedDict(:x => JuMP.Containers.DenseAxisArray(lbs[end:-1:1], 2:-1:1))
        @test getmax(dict_model) ==
            OrderedDict(:x => JuMP.Containers.DenseAxisArray(ubs[end:-1:1], 2:-1:1))
        @test obj(x) == 12 * x[:x][1]
        @test ineq(x) == [
            100 - 6 * x[:x][1] - 8 * x[:x][2],
            120 - 7 * x[:x][1] - 12 * x[:x][2],
            x[:x][1] + x[:x][2] - 50,
        ]
        @test eq(x) == [x[:x][1] + x[:x][2] - 25]

        vec_model, _, _ = NonconvexCore.tovecmodel(dict_model)
        vec_x = NonconvexCore.getinit(vec_model)
        vec_obj = NonconvexCore.getobjective(vec_model)
        vec_ineq = NonconvexCore.getineqconstraints(vec_model)
        vec_eq = NonconvexCore.geteqconstraints(vec_model)

        val0, grad0 = NonconvexCore.value_gradient(vec_obj, vec_x)
        val1, jac1 = NonconvexCore.value_jacobian(vec_ineq, vec_x)
        val2, jac2 = NonconvexCore.value_jacobian(vec_eq, vec_x)

        @test val0 == obj(x)
        @test val1 == ineq(x)
        @test val2 == eq(x)

        @test grad0 == [0, 12]
        @test jac1 == [-8 -6; -12 -7; 1 1]
        @test jac2 == [1 1;]
    end

    @testset "nonlinear JuMP model" begin
        model = JuMP.Model()
        JuMP.@variable(model, x[1:2] >= 0)
        JuMP.@objective(model, Min, x[1]^2 + sin(x[2]))
        JuMP.@constraint(model, sin(x[1]) + x[2]^3 >= 1)
        JuMP.@constraint(model, x[1] + cos(x[2]) == 2)
        JuMP.@constraint(model, -2 <= sin(x[1]) + x[2] <= 3)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        @test obj(x0) == x0[:x][1]^2 + sin(x0[:x][2])
        @test ineq(x0) == [
            1 - sin(x0[:x][1]) - x0[:x][2]^3,
            -2 - sin(x0[:x][1]) - x0[:x][2],
            sin(x0[:x][1]) + x0[:x][2] - 3,
        ]
        @test eq(x0) == [x0[:x][1] + cos(x0[:x][2]) - 2]

        # Derivatives flow through the Nonlinear evaluator's rrule (reverse-mode
        # backends that consult ChainRules).
        vm, xv, _ = NonconvexCore.tovecmodel(dict_model)
        v, g = NonconvexCore.value_gradient(vm.objective, xv, ADTypes.AutoZygote())
        @test g ≈ [2xv[1], cos(xv[2])]
        _, jac = NonconvexCore.value_jacobian(vm.ineq_constraints, xv, ADTypes.AutoZygote())
        @test Matrix(jac) ≈ [
            -cos(xv[1]) -3xv[2]^2
            -cos(xv[1]) -1
            cos(xv[1]) 1
        ]
        _, jaceq = NonconvexCore.value_jacobian(vm.eq_constraints, xv, ADTypes.AutoZygote())
        @test Matrix(jaceq) ≈ [1 -sin(xv[2])]

        # The sparse path uses the graph structure and must include the interval
        # rows.
        J = NonconvexCore.sparse_jacobian(vm.ineq_constraints, xv, ADTypes.AutoZygote())
        @test Matrix(J) ≈ Matrix(jac)
        @test size(J) == (3, 2)
    end

    @testset "x[(ind1, ind2)] variable names" begin
        model = JuMP.Model()
        lbs = [0.0, 0.0]
        ubs = [Inf, 3.0]
        inds = [(1, 1), (2, 1)]
        JuMP.@variable(
            model, x[ind = inds], lower_bound=lbs[ind[1]], upper_bound=ubs[ind[1]]
        )
        JuMP.@objective(model, Min, 12x[(1, 1)] + 20x[(2, 1)])
        JuMP.@constraint(model, c1, 6x[(1, 1)] + 8x[(2, 1)] >= 100)
        JuMP.@constraint(model, c2, 7x[(1, 1)] + 12x[(2, 1)] >= 120)
        JuMP.@constraint(model, c3, x[(1, 1)] + x[(2, 1)] == 25)
        JuMP.@constraint(model, c4, x[(1, 1)] + x[(2, 1)] <= 50)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x = NonconvexCore.getinit(dict_model)

        @test getmin(dict_model) ==
            OrderedDict(:x => JuMP.Containers.DenseAxisArray(lbs, [(1, 1), (2, 1)]))
        @test getmax(dict_model) ==
            OrderedDict(:x => JuMP.Containers.DenseAxisArray(ubs, [(1, 1), (2, 1)]))
        @test obj(x) == 12 * x[:x][(1, 1)] + 20 * x[:x][(2, 1)]
        @test ineq(x) == [
            100 - 6 * x[:x][(1, 1)] - 8 * x[:x][(2, 1)],
            120 - 7 * x[:x][(1, 1)] - 12 * x[:x][(2, 1)],
            x[:x][(1, 1)] + x[:x][(2, 1)] - 50,
        ]
        @test eq(x) == [x[:x][(1, 1)] + x[:x][(2, 1)] - 25]

        vec_model, _, _ = NonconvexCore.tovecmodel(dict_model)
        vec_x = NonconvexCore.getinit(vec_model)
        vec_obj = NonconvexCore.getobjective(vec_model)
        vec_ineq = NonconvexCore.getineqconstraints(vec_model)
        vec_eq = NonconvexCore.geteqconstraints(vec_model)

        val0, grad0 = NonconvexCore.value_gradient(vec_obj, vec_x)
        val1, jac1 = NonconvexCore.value_jacobian(vec_ineq, vec_x)
        val2, jac2 = NonconvexCore.value_jacobian(vec_eq, vec_x)

        @test val0 == obj(x)
        @test val1 == ineq(x)
        @test val2 == eq(x)

        @test grad0 == [12, 20]
        @test jac1 == [-6 -8; -7 -12; 1 1]
        @test jac2 == [1 1;]
    end

    @testset "Quadratic objective" begin
        model = JuMP.Model()
        JuMP.@variable(model, x >= 0)
        JuMP.@variable(model, y >= 0)
        JuMP.@objective(model, Min, x^2 + 2x * y + 3y^2 + x + 2y)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        p = OrderedDict(:x => 1.0, :y => 2.0)
        @test obj(p) == 1.0 + 2 * 1.0 * 2.0 + 3 * 4.0 + 1.0 + 4.0

        vec_model, _, _ = NonconvexCore.tovecmodel(dict_model)
        vec_obj = NonconvexCore.getobjective(vec_model)
        val, grad = NonconvexCore.value_gradient(vec_obj, [1.0, 2.0])
        @test val == 22.0
        @test grad == [2 * 1.0 + 2 * 2.0 + 1.0, 2 * 1.0 + 6 * 2.0 + 2.0]
    end

    @testset "Quadratic objective with max sense" begin
        model = JuMP.Model()
        JuMP.@variable(model, x >= 0)
        JuMP.@objective(model, Max, x^2 + 2x)
        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        @test obj(OrderedDict(:x => 3.0)) == -(9.0 + 6.0)
    end

    @testset "Nonlinear objective conversion" begin
        model = JuMP.Model()
        JuMP.@variable(model, x >= 0)
        JuMP.@objective(model, Min, x^3)
        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        @test obj(x0) == x0[:x]^3
    end

    @testset "Closed-form channel: affine and quadratic rows" begin
        # Quadratic objective, affine and quadratic inequality rows, and a
        # nonlinear equality row: the affine/quadratic parts must be evaluated
        # in closed form (out of the Nonlinear evaluator).
        model = JuMP.Model()
        JuMP.@variable(model, x[1:3])
        JuMP.@objective(model, Min, 2x[1]^2 + x[1] * x[2] + 3x[2] + x[3])
        JuMP.@constraint(model, x[1] + 2x[2] - x[3] >= 1)
        JuMP.@constraint(model, x[2]^2 + x[3]^2 <= 2)
        JuMP.@constraint(model, sin(x[1]) + x[2] == 0.5)

        dict_model = DictModel(model)
        obj = NonconvexCore.getobjective(dict_model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        xv = NonconvexCore.flatten(x0)[1]

        # Values.
        @test obj(x0) ≈ 2xv[1]^2 + xv[1] * xv[2] + 3xv[2] + xv[3]
        @test ineq(x0) ≈ [1 - xv[1] - 2xv[2] + xv[3], xv[2]^2 + xv[3]^2 - 2]
        @test eq(x0) ≈ [sin(xv[1]) + xv[2] - 0.5]

        # The evaluator holds only the nonlinear row.
        objraw = obj.f
        m = objraw.m
        @test length(m.ev.ordered_constraints) == 1

        vm, _, _ = NonconvexCore.tovecmodel(dict_model)
        v, g = NonconvexCore.value_gradient(vm.objective, xv)
        @test v ≈ obj(x0)
        @test g ≈ [4xv[1] + xv[2], xv[1] + 3, 1]

        _, Jineq = NonconvexCore.value_jacobian(vm.ineq_constraints, xv)
        @test Matrix(Jineq) ≈ [-1 -2 1; 0 2xv[2] 2xv[3]]

        _, Jeq = NonconvexCore.value_jacobian(vm.eq_constraints, xv)
        @test Matrix(Jeq) ≈ [cos(xv[1]) 1 0]

        # The sparse Jacobian of the merged channel matches the dense one.
        Jsp = NonconvexCore.sparse_jacobian(vm.ineq_constraints, xv)
        @test Matrix(Jsp) == Matrix(Jineq)

        # Lagrangian Hessian: σ∇²f + λ2∇²g2 + λ3∇²g3 (the affine row adds
        # nothing) with all multipliers and the objective factor set to one.
        H = zeros(3, 3)
        hvals = zeros(length(m.hess_rows))
        NonconvexCore.jump_hessian_lagrangian!(m, hvals, xv, 1.0, ones(3))
        for (k, (i, j)) in enumerate(zip(m.hess_rows, m.hess_cols))
            H[i, j] = hvals[k]
            H[j, i] = hvals[k]
        end
        Hexact = [4 - sin(xv[1]) 1 0; 1 2 0; 0 0 2]
        @test H ≈ Hexact
    end

    @testset "Closed-form channel: interval and equality rows" begin
        model = JuMP.Model()
        JuMP.@variable(model, x[1:2])
        JuMP.@objective(model, Min, x[1]^2)
        JuMP.@constraint(model, -1 <= x[1] + x[2] <= 2)
        JuMP.@constraint(model, x[1] * x[2] + x[2] == 0)
        # A nonlinear row forces the expression path (the matrix path does not
        # cover anonymous or interval constraints).
        JuMP.@constraint(model, sin(x[1]) >= -1.5)
        dict_model = DictModel(model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        eq = NonconvexCore.geteqconstraints(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        xv = NonconvexCore.flatten(x0)[1]
        # The converted rows follow `all_constraints`, which lists the
        # nonlinear block rows before the scalar affine rows.
        @test ineq(x0) == [-1.5 - sin(xv[1]), -1 - xv[1] - xv[2], xv[1] + xv[2] - 2]
        @test eq(x0) == [xv[1] * xv[2] + xv[2]]
        vm, _, _ = NonconvexCore.tovecmodel(dict_model)
        _, Jineq = NonconvexCore.value_jacobian(vm.ineq_constraints, xv)
        @test Matrix(Jineq) == [-cos(xv[1]) 0; -1 -1; 1 1]
        _, Jeq = NonconvexCore.value_jacobian(vm.eq_constraints, xv)
        @test Matrix(Jeq) == [xv[2] xv[1] + 1]
    end

    @testset "Nonlinear objective Hessian survives affine-only constraints" begin
        # Regression: the objective Hessian lives only in the Nonlinear
        # evaluator; it must not be dropped when no constraint row is nonlinear.
        model = JuMP.Model()
        JuMP.@variable(model, x >= 0)
        JuMP.@variable(model, y >= 0)
        JuMP.@objective(model, Min, x^3 + y)
        JuMP.@constraint(model, x + 2y <= 1)  # affine only
        dict_model = DictModel(model)
        vm, _, _ = NonconvexCore.tovecmodel(dict_model)
        xv = [1.0, 1.0]
        obj = NonconvexCore.getobjective(vm)
        jf = NonconvexCore.getrawfunction(NonconvexCore.getfunction(obj))
        m = jf.m
        hvals = zeros(length(m.hess_rows))
        NonconvexCore.jump_hessian_lagrangian!(m, hvals, xv, 1.0, Float64[])
        H = zeros(2, 2)
        for (k, (i, j)) in enumerate(zip(m.hess_rows, m.hess_cols))
            H[i, j] = hvals[k]
            H[j, i] = hvals[k]
        end
        @test H ≈ [6.0 0.0; 0.0 0.0]
        # The higher-level API reads the same evaluator path.
        H2 = NonconvexCore.value_hessian(NonconvexCore.getfunction(obj), xv)
        @test H2 ≈ [6.0 0.0; 0.0 0.0]
    end

    @testset "Merged Hessian structure is lower-triangular" begin
        # Regression: closed-form (quadratic) Hessian entries must be stored in
        # the same lower-triangle convention as the MOI evaluator's entries
        # (row >= col), so cross-channel entries dedupe and Ipopt's eval_h
        # receives a valid structure.
        model = JuMP.Model()
        JuMP.@variable(model, y[1:2])
        JuMP.@objective(model, Min, y[1] * y[2])  # closed-form objective, term (1, 2)
        JuMP.@constraint(model, y[1]^2 * y[2] / 2 >= 1)  # nonlinear row
        dict_model = DictModel(model)
        objraw = NonconvexCore.getobjective(dict_model).f
        m = objraw.m
        @test all(i -> i[1] >= i[2], zip(m.hess_rows, m.hess_cols))
        @test !any(((i, j),) -> i < j, zip(m.hess_rows, m.hess_cols))
        # The objective-only Hessian (zero multipliers) is just ∇²(y₁y₂).
        xv = [2.0, 3.0]
        hvals = zeros(length(m.hess_rows))
        NonconvexCore.jump_hessian_lagrangian!(m, hvals, xv, 1.0, zeros(1))
        H = zeros(2, 2)
        for (k, (i, j)) in enumerate(zip(m.hess_rows, m.hess_cols))
            H[i, j] = hvals[k]
            H[j, i] = hvals[k]
        end
        @test H ≈ [0.0 1.0; 1.0 0.0]
    end

    @testset "Closed-form constraint Hessian applies the row sign" begin
        # Regression: a `>=` quadratic row stores its value as `sign*f + shift`
        # with sign = -1, so its Hessian contribution must carry the same sign.
        model = JuMP.Model()
        JuMP.@variable(model, z[1:2])
        JuMP.@constraint(model, z[1]^2 + z[2]^2 >= 1)  # closed-form, sign -1
        JuMP.@constraint(model, z[1]^3 + z[2] == 0)    # nonlinear row (expression path)
        dict_model = DictModel(model)
        ineqraw = NonconvexCore.getineqconstraints(dict_model).fs[1].f
        m = NonconvexCore.getfunction(ineqraw).m
        xv = [1.0, 1.0]
        hvals = zeros(length(m.hess_rows))
        # λ = [1, 0]: only the `>=` quadratic row contributes.
        NonconvexCore.jump_hessian_lagrangian!(m, hvals, xv, 0.0, [1.0, 0.0])
        H = zeros(2, 2)
        for (k, (i, j)) in enumerate(zip(m.hess_rows, m.hess_cols))
            H[i, j] = hvals[k]
            H[j, i] = hvals[k]
        end
        @test H ≈ [-2.0 0.0; 0.0 -2.0]
    end

    @testset "Quadratic constraints via DictModel" begin
        # Regression: named quadratic constraints used to crash the matrix
        # path with `AssertionError: func isa AffExpr`, and anonymous ones
        # were silently dropped. Both now take the expression path.
        model = JuMP.Model()
        JuMP.@variable(model, u[1:2])
        JuMP.@constraint(model, cq, u[1]^2 + u[2]^2 <= 1)   # named quadratic
        JuMP.@constraint(model, u[1]^2 + u[2]^2 >= 0.5)     # anonymous quadratic
        JuMP.@objective(model, Min, u[1] + u[2])
        dict_model = DictModel(model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        xv = NonconvexCore.flatten(x0)[1]
        # One expression-path evaluator covers every row; the rows follow
        # `all_constraints`, which lists them by set type rather than by
        # insertion order, so compare the values as a set.
        @test length(ineq.fs) == 1
        vals = ineq(x0)
        @test length(vals) == 2  # neither constraint dropped
        @test sort(vals) ≈ sort([xv[1]^2 + xv[2]^2 - 1, 0.5 - xv[1]^2 - xv[2]^2])
    end

    @testset "Anonymous affine constraints via DictModel" begin
        # Regression: the matrix path iterated `model.obj_dict` and silently
        # dropped anonymous constraints.
        model = JuMP.Model()
        JuMP.@variable(model, v[1:2])
        JuMP.@objective(model, Min, v[1] + v[2])
        JuMP.@constraint(model, v[1] + v[2] <= 1)  # anonymous affine
        dict_model = DictModel(model)
        ineq = NonconvexCore.getineqconstraints(dict_model)
        x0 = NonconvexCore.getinit(dict_model)
        @test length(ineq.fs) == 1
        @test ineq(x0) ≈ [sum(sum(x0[k]) for k in keys(x0)) - 1]
    end

    @testset "value_hessian on JuMP-converted objectives" begin
        # Regression: prepare_model/hessian and value_hessian used to route
        # JuMP-converted objectives through generic AD (crashing in the MOI
        # evaluator); they now read the evaluator's Hessian directly.
        model = JuMP.Model()
        JuMP.@variable(model, w[1:2])
        JuMP.@objective(model, Min, w[1]^2 + w[1] * w[2] + w[2]^2)
        JuMP.@constraint(model, w[1] + w[2] <= 1)
        dict_model = DictModel(model)
        vm, xv, _ = NonconvexCore.tovecmodel(dict_model)
        xv = [1.0, 2.0]
        vm2 = NonconvexCore.prepare_model(vm, xv)  # hessian=true by default
        H = NonconvexCore.value_hessian(NonconvexCore.getobjective(vm2), xv)
        @test H ≈ [2.0 1.0; 1.0 2.0]
        Hsp = NonconvexCore.sparse_hessian(NonconvexCore.getobjective(vm2), xv)
        @test Matrix(Hsp) ≈ [2.0 1.0; 1.0 2.0]
    end

    @testset "Integer variable typing in get_jump_problem" begin
        # Regression: the binary-vs-integer heuristic used to classify any
        # integer variable with bounds inside (-1, 2) as binary (ZeroOne),
        # forcing legitimate integer domains like [-0.5, 0.5] or [0.5, 1.5]
        # into {0, 1}. Only the exact box [0, 1] is binary.
        model = Model()
        set_objective!(model, x -> x[1] + x[2] + x[3])
        addvar!(model, [-0.5, 0.0, 0.5], [0.5, 1.0, 1.5]; integer=[true, true, true])
        vm, xv, _ = NonconvexCore.tovecmodel(model)
        opt = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        jp = NonconvexCore.get_jump_problem(
            NonconvexCore.CountingFunction(NonconvexCore.getobjective(vm)),
            nothing,
            nothing,
            xv,
            vm.integer,
            getmin(vm),
            getmax(vm),
            true,
            () -> opt,
            NonconvexCore.DEFAULT_AD_BACKEND[],
        )
        moi_model = JuMP.backend(jp.model)
        binary_ci = MOI.get(
            moi_model, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}()
        )
        integer_ci = MOI.get(
            moi_model, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}()
        )
        @test length(binary_ci) == 1  # only the [0, 1] variable
        @test length(integer_ci) == 2 # the [-0.5, 0.5] and [0.5, 1.5] variables
    end
end
