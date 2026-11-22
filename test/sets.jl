@testitem "Sets: constructors, dimensions and validation" begin
    using NonconvexCore, LinearAlgebra, Test
    using NonconvexCore: getdim, components, check_set_output

    # Dimensions follow the MOI convention.
    @test getdim(EqualTo(0.0)) == 1
    @test getdim(EqualTo([1.0, 2.0])) == 2
    @test getdim(LessThan(0.0)) == 1
    @test getdim(LessThan([0.0, 0.0])) == 2
    @test getdim(GreaterThan(1.0)) == 1
    @test getdim(Interval(0.0, 1.0)) == 1
    @test getdim(Interval([0.0, 0.0], [1.0, 2.0])) == 2
    @test getdim(Reals(4)) == 4
    @test getdim(Nonnegatives(2)) == 2
    @test getdim(Nonpositives(2)) == 2
    @test getdim(Zeros(2)) == 2
    @test getdim(Simplex(3)) == 3
    sp = SetProduct(LessThan(0.0), EqualTo(1.0))
    @test getdim(sp) == 2
    @test components(sp) == (LessThan(0.0), EqualTo(1.0))
    @test components(LessThan(0.0)) == (LessThan(0.0),)

    # Aliases of the legacy default sets.
    @test InequalitySet === LessThan
    @test EqualitySet === EqualTo
    @test InequalitySet(0.0) isa LessThan

    # Set composition: a vector of simplexes, nesting.
    @test SetProduct(Simplex(4), Simplex(4)).sets isa Tuple{Simplex,Simplex}
    @test getdim(SetProduct(Simplex(4), Simplex(4))) == 8
    nested = SetProduct(LessThan(0.0), SetProduct(GreaterThan(0.0), EqualTo(1.0)))
    @test getdim(nested) == 3

    # Interval bounds must agree in length.
    @test_throws DimensionMismatch Interval([0.0], [1.0, 2.0])

    # Output-shape validation (scalar/vector/product).
    @test_throws ArgumentError check_set_output([1.0], LessThan(0.0))
    @test_throws ArgumentError check_set_output(1.0, LessThan([0.0, 0.0]))
    @test_throws ArgumentError check_set_output(1.0, EqualTo([1.0]))  # length-1 vector bound
    @test_throws ArgumentError check_set_output((1.0,), sp)
    check_set_output((1.0, 1.0), sp)
    check_set_output([1.0, 2.0], EqualTo([1.0, 2.0]))
end

@testitem "Sets: violations" begin
    using NonconvexCore, LinearAlgebra, Test
    using NonconvexCore: set_value

    c1 = FunctionInSet(x -> x[1] - 2.0, LessThan(0.0))
    @test set_violation(c1, [1.0]) == [0.0]
    @test set_violation(c1, [3.0]) == [1.0]
    c2 = FunctionInSet(x -> x[1] - 2.0, GreaterThan(0.0))
    @test set_violation(c2, [3.0]) == [0.0]
    @test set_violation(c2, [1.0]) == [1.0]
    c3 = FunctionInSet(x -> [x[1] - 2.0, x[2] + 1.0], EqualTo([0.0, 0.0]))
    @test set_violation(c3, [2.0, -1.0]) == [0.0, 0.0]
    @test set_violation(c3, [2.5, -1.0]) == [0.5, 0.0]
    c4 = FunctionInSet(x -> [x[1], x[2]], Interval([0.0, 0.0], [1.0, 1.0]))
    @test set_violation(c4, [0.5, 0.5]) == [0.0, 0.0]
    @test set_violation(c4, [1.5, -0.5]) == [0.5, 0.5]
    c6 = FunctionInSet(x -> [x[1], x[2]], Nonnegatives(2))
    @test set_violation(c6, [1.0, 1.0]) == [0.0, 0.0]
    @test set_violation(c6, [-1.0, 1.0]) == [1.0, 0.0]
    c7 = FunctionInSet(x -> [x[1], x[2]], Nonpositives(2))
    @test set_violation(c7, [-1.0, -1.0]) == [0.0, 0.0]
    @test set_violation(c7, [1.0, -1.0]) == [1.0, 0.0]
    c8 = FunctionInSet(x -> [x[1], x[2]], Zeros(2))
    @test set_violation(c8, [0.0, 0.0]) == [0.0, 0.0]
    @test set_violation(c8, [1.0, 0.0]) == [1.0, 0.0]
    c9 = FunctionInSet(x -> [x[1], x[2]], Simplex(2))
    @test set_violation(c9, [0.5, 0.5]) == [0.0, 0.0, 0.0]
    v9 = set_violation(c9, [0.5, 1.0])
    @test v9 ≈ [0.0, 0.0, 0.5]
    # Nested products concatenate their violations.
    c10 = FunctionInSet(
        x -> (x[1] - 1.0, (x[2] - 1.0, x[3])),
        SetProduct(LessThan(0.0), SetProduct(GreaterThan(0.0), EqualTo(2.0))),
    )
    @test set_violation(c10, [0.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    @test set_violation(c10, [2.0, 0.0, 0.0]) == [1.0, 1.0, 2.0]
    # A non-tuple output against a single set is wrapped in a 1-tuple.
    v = set_value(c1, [1.0])
    @test v isa Tuple && length(v) == 1
end

@testitem "FunctionInSet: model registration, AD and flags" begin
    using NonconvexCore, LinearAlgebra, Test, Zygote
    using NonconvexCore:
        getnsetconstraints,
        getnconstraints,
        isfeasible,
        VecModel,
        Objective,
        VectorOfFunctions,
        IneqConstraint,
        EqConstraint

    m = Model(x -> x[1] + x[2])
    addvar!(m, [0.0, 0.0], [5.0, 5.0])
    f = x -> (x[1] - 2.0, x[2] - 1.0)
    add_set_constraint!(m, f, SetProduct(LessThan(0.0), LessThan(0.0)); flags=[:stab])
    @test getnsetconstraints(m) == 1
    @test getnconstraints(m) == 1
    @test m.set_constraints.fs[1].flags == Set([:stab])

    # Registration-time validation catches shape mismatches.
    m2 = Model(x -> x[1])
    addvar!(m2, [0.0], [1.0])
    @test_throws ArgumentError add_set_constraint!(
        m2, x -> ([1.0],), SetProduct(LessThan(0.0))
    )

    vm, x0, unflatten = NonconvexCore.tovecmodel(m, [0.0, 0.0])
    c = vm.set_constraints.fs[1]
    @test c isa FunctionInSet
    out = c(x0)
    @test out isa Tuple
    @test out[1] isa Real
    @test out[2] isa Real

    # Gradients of the routed components.
    @test Zygote.gradient(x -> c(x)[1], [1.0, 1.0])[1] == [1.0, 0.0]
    @test Zygote.gradient(x -> c(x)[2], [1.0, 1.0])[1] == [0.0, 1.0]

    # isfeasible includes set constraints.
    @test isfeasible(vm, [2.0, 1.0])
    @test !isfeasible(vm, [3.0, 1.0])
    @test !isfeasible(vm, [2.0, 2.0])

    # Backward-compatible VecModel constructor defaults to no set constraints.
    vm2 = VecModel(
        Objective(x -> 0.0),
        VectorOfFunctions(()),
        VectorOfFunctions(()),
        VectorOfFunctions(()),
        [0.0],
        [1.0],
        [0.5],
        falses(1),
    )
    @test length(vm2.set_constraints.fs) == 0
    @test isempty(vm2.set_variables)

    # setof/tosetconstraint migration path of the legacy syntax.
    ci = IneqConstraint(x -> x[1], [2.0])
    @test setof(ci) == LessThan([2.0])
    @test tosetconstraint(ci).set == LessThan([2.0])
    @test tosetconstraint(EqConstraint(x -> x[1], [1.0])).set == EqualTo([1.0])
    # The semidefinite conversion is native (PSDCone is a Core set).
    @test setof(NonconvexCore.SDConstraint(x -> [x[1];;], 1)) == PSDCone(1)
    @test tosetconstraint(NonconvexCore.SDConstraint(x -> [x[1];;], 1)).set == PSDCone(1)
end

@testitem "Variables in sets: blocks, labels, synthetics" begin
    using NonconvexCore, LinearAlgebra, Test
    using NonconvexCore: getsetvariables

    m = Model(x -> sum(x))
    add_set_variable!(m, Simplex(3))
    add_set_variable!(m, Reals(2))
    add_set_variable!(m, Nonnegatives(2); flags=[:p])
    @test length(m.box_min) == 7
    vs = getsetvariables(m)
    @test vs[1].range == 1:3
    @test vs[1].set isa Simplex
    @test vs[2].range == 4:5
    @test vs[2].set isa Reals
    @test vs[3].range == 6:7
    @test vs[3].set isa Nonnegatives
    @test vs[3].flags == Set([:p])
    # Default inits: uniform simplex, free zeros, ones.
    @test m.init[1:3] ≈ fill(1 / 3, 3)
    @test m.init[4:5] == zeros(2)
    @test m.init[6:7] == ones(2)
    # Boxes: simplex [0, ∞), reals free, nonnegatives [0, ∞).
    @test m.box_min[1:3] == zeros(3) && m.box_max[1:3] == fill(Inf, 3)
    @test m.box_min[4:5] == fill(-Inf, 2) && m.box_max[4:5] == fill(Inf, 2)
    @test m.box_max[6:7] == fill(Inf, 2)

    # Synthetic constraints: the sum-to-one equality of the simplex block.
    vm, x0, _ = NonconvexCore.tovecmodel(m, m.init)
    @test length(vm.set_constraints.fs) == 1
    @test vm.set_constraints.fs[1].set == EqualTo(1.0)
    @test vm.set_constraints.fs[1](x0) ≈ 1.0

    # The default addvar! falls back on the Interval vector set.
    m2 = Model(x -> x[1])
    addvar!(m2, 0.0, 1.0)
    addvar!(m2, [0.0, 0.0], [1.0, 2.0]; flags=:design)
    vs2 = getsetvariables(m2)
    @test vs2[1].set == Interval(0.0, 1.0)
    @test vs2[1].range == 1:1
    @test vs2[2].set == Interval([0.0, 0.0], [1.0, 2.0])
    @test vs2[2].range == 2:3
    @test vs2[2].flags == Set([:design])
    # Vector-set variables with length-1 vector bounds.
    m3 = Model(x -> x[1])
    add_set_variable!(m3, EqualTo([2.0]))
    @test m3.box_min == [2.0] && m3.box_max == [2.0]
    @test m3.init == [2.0]
end

@testitem "Granular capability declaration for sets" begin
    using NonconvexCore, Test
    using NonconvexCore: set_kinds, variable_set_kinds

    struct SetsCapableAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::SetsCapableAlg) = ConstraintCapabilities(;
        sets_eq=true,
        variable_sets_simplex=true,
        box=true,
        continuous=true,
        infinite_bounds=true,
    )
    struct SetsUnsupportedAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::SetsUnsupportedAlg) = ConstraintCapabilities(;
        sets_eq=true, box=true, continuous=true, infinite_bounds=true
    )
    struct UndeclaredAlg <: NonconvexCore.AbstractOptimizer end

    m = Model(x -> sum(x))
    add_set_variable!(m, Simplex(3))
    vm, x0, _ = NonconvexCore.tovecmodel(m, m.init)

    # A solver missing the simplex-variable capability is rejected.
    @test_throws ArgumentError NonconvexCore.checksubproblem!(vm, SetsUnsupportedAlg())
    # A solver without any declaration must declare capabilities.
    @test_throws ArgumentError NonconvexCore.checksubproblem!(vm, UndeclaredAlg())
    # Explicit support passes.
    NonconvexCore.checksubproblem!(vm, SetsCapableAlg())

    # Defaults: all set capabilities default to false.
    caps = ConstraintCapabilities()
    @test caps.sets_eq === false
    @test caps.sets_ineq === false
    @test caps.sets_sd === false
    @test caps.sets_simplex === false
    @test caps.sets_product === false
    @test caps.sets_chance === false
    @test caps.variable_sets_sd === false
    @test caps.variable_sets_simplex === false
    @test ConstraintCapabilities(sets_chance=true).sets_chance === true
    # merge_capabilities forwards them.
    @test merge_capabilities(ConstraintCapabilities(sets_sd=true)).sets_sd === true
    @test merge_capabilities(nothing; sets_chance=true).sets_chance === true

    # Set kinds used for feature detection (the :sd kind itself is provided
    # by NonconvexSemidefinite, where PSDCone lives).
    @test set_kinds(EqualTo(0.0)) == [:eq]
    @test set_kinds(Zeros(2)) == [:eq]
    @test set_kinds(LessThan(0.0)) == [:ineq]
    @test set_kinds(Interval(0.0, 1.0)) == [:ineq]
    @test set_kinds(Nonnegatives(2)) == [:ineq]
    @test set_kinds(Simplex(2)) == [:simplex]
    @test set_kinds(Reals(2)) == Symbol[]
    @test set_kinds(SetProduct(LessThan(0.0), Zeros(2))) == [:product, :ineq, :eq]
    @test variable_set_kinds(Simplex(2)) == [:simplex]
    @test variable_set_kinds(Interval(0.0, 1.0)) == Symbol[]
    @test variable_set_kinds(SetProduct(Simplex(2), Interval(0.0, 1.0))) == [:simplex]
end

@testitem "Epigraph transform" begin
    using NonconvexCore, Test

    m = Model(x -> (x[1] - 3)^2)
    addvar!(m, 0.0, 5.0)
    blk = epigraph!(m)
    @test blk.set == Interval(-Inf, Inf)
    @test blk.range == 2:2
    # The objective is t; the epigraph inequality f(x) - t ≤ 0.
    @test m.objective.f([1.0, 4.0]) == 4.0
    c = m.set_constraints.fs[1]
    @test c.set == LessThan(0.0)
    @test c([1.0, 4.0]) == 0.0
    @test set_violation(c, [1.0, 3.0]) == [1.0]

    # add_epigraph_variable!: the primitive without the constraint.
    m2 = Model(x -> x[1]^2)
    addvar!(m2, 0.0, 2.0)
    blk2, f, n_old = add_epigraph_variable!(m2; margin=2.0, t_lb=0.0)
    @test n_old == 1
    @test blk2.set == Interval(0.0, Inf)
    @test m2.objective.f([1.0, 3.0]) == 3.0
    @test isempty(m2.set_constraints.fs)
    @test f([1.0]) == 1.0

    # Errors informatively without an objective.
    m3 = Model(nothing)
    addvar!(m3, 0.0, 1.0)
    @test_throws ArgumentError epigraph!(m3)
    @test_throws ArgumentError epigraph!(NonconvexCore.DictModel(nothing))
end

@testitem "Linear constraints and the nullspace transform" begin
    using NonconvexCore, LinearAlgebra, Test, Zygote
    using NonconvexCore: getdim, linear_offset

    # First-class linear function.
    lf = LinearFunction([1.0 2.0; 3.0 4.0], [1.0, 2.0])
    @test lf([1.0, 1.0]) == [4.0, 9.0]
    @test getdim(lf) == 2
    @test LinearFunction([1.0 2.0])([1.0, 1.0]) == [3.0]
    @test linear_offset(LinearFunction([1.0 2.0])) == zeros(1)

    # Registration and exact linearity classification.
    m = Model(x -> sum(abs2, x .- [1.0, 2.0, 3.0]))
    addvar!(m, [-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])
    add_linear_constraint!(m, [1.0 1.0 1.0], EqualTo([6.0]); flags=[:eq1])
    vm, x3, _ = NonconvexCore.tovecmodel(m, [0.0, 0.0, 0.0])
    @test vm.set_constraints.fs[1]([1.0, 2.0, 3.0]) ≈ [6.0]
    @test vm.set_constraints.fs[1].flags == Set([:eq1])
    lin = NonconvexCore._constraint_linearity(vm.set_constraints, x3)
    @test lin == BitVector([true])

    # The nullspace transform eliminates the equality, converts the box into
    # linear inequalities, and maps solutions back.
    rm, z0, mapz = nullspace_transform(vm)
    @test length(rm.box_min) == 2
    @test rm.objective.f([0.0, 0.0]) ≈ m.objective.f([2.0, 2.0, 2.0])
    @test mapz([0.0, 0.0]) ≈ [2.0, 2.0, 2.0]
    @test length(rm.ineq_constraints.fs) == 2
    c1 = rm.ineq_constraints.fs[1]
    @test c1.f isa LinearFunction
    @test all(c1.f([0.0, 0.0]) .<= 0)
    # The affine map satisfies the eliminated equality for any z.
    for z in ([0.3, -1.2], [5.0, 5.0])
        @test ([1.0 1.0 1.0] * mapz(z))[1] ≈ 6.0
    end
    # Gradient equivalence through the map: ∇_z f = J(z)' ∇_x f.
    using ForwardDiff
    zt = [0.1, 0.2]
    gz = Zygote.gradient(z -> rm.objective.f(z), zt)[1]
    gx = Zygote.gradient(x -> m.objective.f(x), mapz(zt))[1]
    J = ForwardDiff.jacobian(mapz, zt)
    @test gz ≈ J' * gx

    # Models without linear equalities come back unchanged with identity map.
    m2 = Model(x -> x[1]^2)
    addvar!(m2, -1.0, 1.0)
    vm2, x0_2, _ = NonconvexCore.tovecmodel(m2, [0.0])
    rm2, z02, map2 = nullspace_transform(vm2)
    @test rm2 === vm2
    @test map2 === identity
end

@testitem "Unconstrained domain transform" begin
    using NonconvexCore, LinearAlgebra, Test, Zygote

    # Box variables selected by labels become unconstrained.
    m = Model(x -> sum(abs2, x .- [0.5, 0.5, 2.0]))
    addvar!(m, [0.0, 0.0], [1.0, 1.0]; init=[0.5, 0.5], flags=[:p])
    addvar!(m, -10.0, 10.0; init=2.0)
    vm, x4, _ = NonconvexCore.tovecmodel(m, [0.5, 0.5, 2.0])
    um, z0, mapu = unconstrained_transform(vm; labels=[:p])
    @test getmin(um)[3] == -10.0 && getmax(um)[3] == 10.0  # unlabeled kept
    @test getmin(um)[1] == -Inf && getmax(um)[2] == Inf    # labeled freed
    @test z0 ≈ [0.0, 0.0, 2.0]
    @test mapu([0.0, 0.0, 2.0]) ≈ [0.5, 0.5, 2.0]
    @test um.objective.f([0.0, 0.0, 2.0]) ≈ 0.0
    # Gradient flows through the TransformVariables bijection (rrule).
    gz = Zygote.gradient(z -> um.objective.f(z), [0.0, 0.0, 2.0])[1]
    @test gz ≈ [0.0, 0.0, 0.0] atol = 1e-6
    gz2 = Zygote.gradient(z -> um.objective.f(z), [1.0, -1.0, 2.0])[1]
    @test all(isfinite, gz2) && norm(gz2[1:2]) > 0

    # A simplex block becomes n-1 unconstrained reals through the Stan
    # log-ratio/softmax transform.
    ms = Model(x -> sum(abs2, x .- [0.25, 0.25, 0.5]))
    add_set_variable!(ms, Simplex(3); init=[0.25, 0.25, 0.5], flags=[:s])
    vms, xs, _ = NonconvexCore.tovecmodel(ms, [0.25, 0.25, 0.5])
    ums, zs, maps = unconstrained_transform(vms; labels=[:s])
    @test length(ums.box_min) == 2
    y = maps(zs)
    @test sum(y) ≈ 1.0 && all(>(0), y) && y ≈ [0.25, 0.25, 0.5]
    gzs = Zygote.gradient(z -> ums.objective.f(z), zs)[1]
    @test length(gzs) == 2 && all(isfinite, gzs)

    # Integer variables are rejected informatively.
    mi = Model(x -> x[1]^2)
    addvar!(mi, 0.0, 1.0; integer=true)
    vmi, xi, _ = NonconvexCore.tovecmodel(mi, [0.0])
    @test_throws ArgumentError unconstrained_transform(vmi)
end

@testitem "Probabilistic (chance) constraints" begin
    using NonconvexCore, LinearAlgebra, Test, Zygote, Random
    using NonconvexCore: IneqConstraint, chance

    # The scenario batch is drawn by the caller (Core does not sample).
    scenarios(rng, n) = [rand(rng) for _ in 1:n]

    # P(x + ξ ≤ 1) ≥ 0.9 with ξ ~ Uniform(0, 1) ⟺ x ≤ 0.1.
    m = Model(x -> x[1]^2)
    addvar!(m, 0.0, 1.0)
    add_chance_constraint!(
        m,
        (x, ξ) -> x[1] + ξ,
        scenarios(MersenneTwister(42), 2000),
        LessThan(1.0);
        epsilon=0.1,
    )
    vm, x5, _ = NonconvexCore.tovecmodel(m, [0.0])
    c = vm.set_constraints.fs[1]
    @test c.set isa ChanceSet
    @test c.set.set == LessThan(1.0)
    @test length(c.set.scenarios) == 2000
    @test set_violation(c, [0.0])[1] ≈ 0.0 atol = 1e-12
    # The tanh-smoothed exceedance: ≈ 0.5 · mean(tanh(5v)) ≈ 0.36 → 0.26.
    vhalf = set_violation(c, [0.5])[1]
    @test 0.2 < vhalf < 0.35
    @test set_violation(c, [0.1])[1] <= 0.01
    # The smoothed exceedance is differentiable.
    g = Zygote.gradient(x -> set_violation(c, x)[1], [0.5])[1]
    @test isfinite(g[1]) && g[1] > 0

    # Per-output probabilities over a product of sets.
    m2 = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m2, [0.0, 0.0], [1.0, 1.0])
    add_chance_constraint!(
        m2,
        (x, ξ) -> (x[1] + ξ, x[2] + ξ),
        scenarios(MersenneTwister(11), 2000),
        SetProduct(LessThan(1.0), LessThan(1.5));
        epsilon=[0.1, 0.3],
    )
    vm2, x6, _ = NonconvexCore.tovecmodel(m2, [0.0, 0.0])
    c2 = vm2.set_constraints.fs[1]
    @test c2.set.epsilon == [0.1, 0.3]
    # At x = (0.6, 0.6): output 1 exceeds with P ≈ 0.6 (smoothed ≈ 0.36
    # excess over ε₁ = 0.1), output 2 is feasible with P = 1.
    v2 = set_violation(c2, [0.6, 0.6])
    @test v2[1] ≈ 0.36 atol = 0.05
    @test v2[2] == 0.0

    # Deterministic constraints with the special value epsilon = Zero().
    m3 = Model(x -> x[1]^2)
    addvar!(m3, 0.0, 1.0)
    add_chance_constraint!(m3, (x, ξ) -> x[1], nothing, LessThan(0.5); epsilon=Zero())
    vm3, x7, _ = NonconvexCore.tovecmodel(m3, [0.0])
    c3 = vm3.set_constraints.fs[1]
    @test c3.set.epsilon isa Zero
    @test set_violation(c3, [0.3]) == [0.0]
    @test set_violation(c3, [0.7]) ≈ [0.2]
    # A probabilistic constraint requires a scenario batch.
    @test_throws ArgumentError ChanceSet(LessThan(0.0), nothing; epsilon=0.1)

    # Any existing constraint type can be turned probabilistic.
    ci = IneqConstraint((x, ξ) -> x[1] + ξ - 1.0, 0.0)  # x + ξ ≤ 1
    pc = chance(ci, scenarios(MersenneTwister(3), 500); epsilon=0.1)
    @test pc isa FunctionInSet
    @test pc.set isa ChanceSet
    @test pc.set.set isa LessThan
    # Interval (two-sided) bound inside a chance constraint.
    m4 = Model(x -> x[1]^2)
    addvar!(m4, 0.0, 1.0)
    add_chance_constraint!(
        m4,
        (x, ξ) -> x[1] + ξ,
        scenarios(MersenneTwister(7), 500),
        Interval(-1.0, 1.0);
        epsilon=0.1,
    )
    @test m4.set_constraints.fs[1].set.set isa Interval
end

@testitem "PSDCone set: dimensions, validation and violations" begin
    using NonconvexCore, LinearAlgebra, Test
    using NonconvexCore: getdim, components, check_set_output, set_value

    # Dimensions follow the MOI convention (PSDCone(n) counts n²).
    @test getdim(PSDCone(3)) == 9
    sp = SetProduct(LessThan(0.0), PSDCone(2), EqualTo(1.0))
    @test getdim(sp) == 6
    @test components(sp) == (LessThan(0.0), PSDCone(2), EqualTo(1.0))

    # Set composition: a vector of PSD matrices, nesting.
    vp = SetProduct([PSDCone(2), PSDCone(2)])
    @test vp isa SetProduct && length(vp.sets) == 2
    @test SetProduct(PSDCone(3), PSDCone(3)).sets isa Tuple{PSDCone,PSDCone}
    nested = SetProduct(LessThan(0.0), SetProduct(PSDCone(2), PSDCone(2)))
    @test getdim(nested) == 1 + 8

    # Output-shape validation (matrix-shaped).
    @test_throws ArgumentError check_set_output([1.0, 2.0, 3.0], PSDCone(2))
    @test_throws ArgumentError check_set_output(zeros(3, 3), PSDCone(2))
    check_set_output((1.0, zeros(2, 2), 1.0), sp)
    check_set_output([zeros(2, 2), zeros(2, 2)], vp)  # vector output for a product

    # Violations: max(0, -λmin(P)).
    c5 = FunctionInSet(x -> [x[1] - 1.0;;], PSDCone(1))
    @test set_violation(c5, [2.0]) == [0.0]
    @test set_violation(c5, [0.0]) == [1.0]
    @test set_violation(c5, [0.5]) == [0.5]
    # Nested products with a PSD component concatenate their violations.
    c10 = FunctionInSet(
        x -> (x[1] - 1.0, (x[2] - 1.0, [x[3] - 1.0;;])),
        SetProduct(LessThan(0.0), SetProduct(GreaterThan(0.0), PSDCone(1))),
    )
    @test set_violation(c10, [0.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    @test set_violation(c10, [2.0, 0.0, 0.0]) == [1.0, 1.0, 1.0]

    # has_psd_component detects PSD components through products.
    @test has_psd_component(PSDCone(2))
    @test !has_psd_component(LessThan(0.0))
    @test has_psd_component(SetProduct(LessThan(0.0), PSDCone(2)))
    @test !has_psd_component(SetProduct(LessThan(0.0), Simplex(2)))

    # The set view of the classic SD constraint (Core's placeholder errors
    # without this package; here it is overwritten).
    sd = NonconvexCore.SDConstraint(x -> [x[1];;], 1)
    @test setof(sd) == PSDCone(1)
    @test tosetconstraint(sd).set == PSDCone(1)
    @test tosetconstraint(sd) isa FunctionInSet
end

@testitem "PSDCone variables: blocks, capabilities and kinds" begin
    using NonconvexCore, LinearAlgebra, Test
    using NonconvexCore: getsetvariables, set_kinds, variable_set_kinds

    # A PSD matrix variable block.
    m = Model(x -> x[1] + sum(x[2:5]))
    add_set_variable!(m, PSDCone(2))
    add_set_variable!(m, Simplex(3))
    vs = getsetvariables(m)
    @test length(m.box_min) == 7
    @test vs[1].range == 1:4
    @test vs[1].set isa PSDCone
    @test vs[2].set isa Simplex
    # Default inits: identity for PSD, uniform simplex.
    @test m.init[1:4] == vec(Matrix{Float64}(I, 2, 2))
    @test m.init[5:7] ≈ fill(1 / 3, 3)
    # Boxes: PSD free, simplex [0, ∞).
    @test m.box_min[1:4] == fill(-Inf, 4) && m.box_max[1:4] == fill(Inf, 4)
    @test m.box_min[5:7] == zeros(3) && m.box_max[5:7] == fill(Inf, 3)
    # Synthetic constraints: one PSD + one sum-to-one equality.
    vm, x0, _ = NonconvexCore.tovecmodel(m, m.init)
    @test length(vm.set_constraints.fs) == 2
    @test vm.set_constraints.fs[1].set == PSDCone(2)
    @test vm.set_constraints.fs[1](x0) ≈ I(2)
    @test vm.set_constraints.fs[2].set == EqualTo(1.0)
    @test vm.set_constraints.fs[2](x0) ≈ 1.0

    # Set kinds used for feature detection.
    @test set_kinds(PSDCone(2)) == [:sd]
    @test set_kinds(SetProduct(LessThan(0.0), PSDCone(2))) == [:product, :ineq, :sd]
    @test variable_set_kinds(PSDCone(2)) == [:sd]
    @test variable_set_kinds(SetProduct(Simplex(2), PSDCone(2))) == [:simplex, :sd]

    # Capability declaration for semidefinite models.
    struct SDCapableAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::SDCapableAlg) = ConstraintCapabilities(;
        sets_eq=true,
        sets_sd=true,
        variable_sets_sd=true,
        variable_sets_simplex=true,
        box=true,
        continuous=true,
        infinite_bounds=true,
    )
    struct SDMissingAlg <: NonconvexCore.AbstractOptimizer end
    NonconvexCore.solver_capabilities(::SDMissingAlg) = ConstraintCapabilities(;
        sets_eq=true,
        variable_sets_simplex=true,
        box=true,
        continuous=true,
        infinite_bounds=true,
    )
    @test_throws ArgumentError NonconvexCore.checksubproblem!(vm, SDMissingAlg())
    NonconvexCore.checksubproblem!(vm, SDCapableAlg())
end

@testitem "Coverage: error paths and back-compat constructors" begin
    using NonconvexCore, LinearAlgebra, Test, Zygote

    # The back-compat 8-argument Model constructor.
    m = Model(
        NonconvexCore.Objective(x -> x[1]^2),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        [0.0],
        [1.0],
        [0.5],
        [false],
    )
    @test length(m.set_constraints.fs) == 0
    @test isempty(m.set_variables)

    # add_chance_constraint! with a classical constraint (the first method).
    ci = NonconvexCore.IneqConstraint((x, ξ) -> x[1] + ξ, 0.0)
    mc = Model(x -> x[1]^2)
    addvar!(mc, 0.0, 1.0)
    add_chance_constraint!(mc, ci, [0.1, 0.2]; epsilon=0.1)
    @test mc.set_constraints.fs[1].set isa ChanceSet

    # add_set_variable! with a wrong-length initial point.
    mw = Model(x -> x[1])
    @test_throws ArgumentError add_set_variable!(mw, Nonnegatives(2); init=[0.5])

    # DictModel rejects set variables informatively.
    md = DictModel(x -> x[:a])
    addvar!(md, :a, 0.0, 1.0)
    @test_throws ArgumentError add_set_variable!(md, :a, Simplex(2))

    # nullspace_transform rejects integer variables informatively.
    m2 = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m2, [0.0, 0.0], [1.0, 1.0]; integer=[true, false])
    add_linear_constraint!(m2, [1.0 1.0], EqualTo([1.0]))
    vm2, x2, _ = NonconvexCore.tovecmodel(m2, [0.0, 0.0])
    @test_throws ArgumentError nullspace_transform(vm2)
    # nullspace_transform on a model without linear equalities is the identity.
    m3 = Model(x -> x[1]^2)
    addvar!(m3, 0.0, 1.0)
    vm3, x3, _ = NonconvexCore.tovecmodel(m3, [0.0])
    rm3, z3, map3 = nullspace_transform(vm3)
    @test rm3 === vm3

    # The TransformVariables rrules: direct Zygote gradients through both
    # directions exercise the pullbacks.
    using TransformVariables: TransformVariables
    const TV = TransformVariables
    t = TV.as(Array, 2)
    z0 = [0.3, -0.2]
    g = Zygote.gradient(z -> sum(TV.transform(t, z)), z0)[1]
    @test all(isfinite, g) && norm(g) > 0
    x0 = TV.transform(t, z0)
    g2 = Zygote.gradient(x -> sum(TV.inverse(t, x)), x0)[1]
    @test all(isfinite, g2) && norm(g2) > 0
end
