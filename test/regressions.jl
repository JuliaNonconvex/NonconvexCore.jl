@testitem "Regressions" begin
using Test
import JuMP
using NonconvexCore:
    Model,
    Solution,
    ConvergenceState,
    getresiduals,
    assess_convergence!,
    correctsolution!,
    set_objective_multiple!,
    ScaledKKTCriteria,
    KKTCriteria,
    IpoptCriteria,
    GenericCriteria,
    Tolerance,
    DictModel,
    tovecmodel,
    getinit,
    getdim,
    addvar!,
    getineqconstraints,
    value_jacobian,
    VectorOfFunctions,
    @params

# Regression tests for bugs found in the static audit.

@testset "Regression tests" begin
    @testset "Constraint-free models" begin
        m = Model(x -> sum(x.^2))
        addvar!(m, [1.0, 1.0], [2.0, 2.0])
        vm, x0, _ = tovecmodel(m)
        # Empty VectorOfFunctions calls return a typed empty vector.
        g = getineqconstraints(vm)(x0)
        @test g isa Vector{Float64}
        @test isempty(g)
        # getresiduals does not crash on empty g (previously threw
        # ArgumentError: cannot construct a value of type Union{}).
        convstate = ConvergenceState{Float64}()
        s = Solution(
            x0 .+ 1,
            x0,
            Float64[],
            2.0,
            3.0,
            [2.0, 2.0],
            g,
            zeros(0, 2),
            convstate,
        )
        for criteria in (GenericCriteria(), KKTCriteria(), IpoptCriteria())
            Δx, Δf, infeas, f = getresiduals(s, vm, criteria)
            @test infeas == 0.0
        end
    end

    @testset "VectorOfFunctions constructor" begin
        vof = VectorOfFunctions(())
        @test isempty(vof([1.0, 2.0]))
        @test getdim(vof) == 0
    end

    @testset "ScaledKKTCriteria with negative objective multiple" begin
        m = Model(x -> x[1]^2 + x[2]^2)
        addvar!(m, [-1.0, -1.0], [1.0, 1.0])
        vm, x0, _ = tovecmodel(m)
        set_objective_multiple!(vm, -1.0)
        convstate = ConvergenceState{Float64}()
        g = getineqconstraints(vm)(x0)
        s = Solution(
            [0.5, 0.5],
            [0.0, 0.0],
            Float64[],
            2.0,
            3.0,
            [100.0, 100.0],
            g,
            zeros(0, 2),
            convstate,
        )
        # A solution far from optimal must not be flagged as converged when
        # the objective multiple is negative (maximization).
        assess_convergence!(
            s,
            vm,
            Tolerance(kkt = 1e-10),
            ScaledKKTCriteria(),
            false,
            1,
        )
        @test !s.convstate.converged
        @test s.convstate.kkt_residual >= 0
    end

    @testset "One-sided bounds in JuMP conversion" begin
        jm = JuMP.Model()
        JuMP.@variable(jm, -Inf <= y <= 5)
        JuMP.@objective(jm, Min, 1.0 * y)
        dm = DictModel(jm)
        # Initial value should be ub - 1 = 4, not lb - 1 = -Inf.
        @test getinit(dm)[:y] == 4.0
    end

    @testset "value_jacobian with plain functions" begin
        val, jac = value_jacobian(x -> [x[1]^2, x[2]], [3.0, 4.0])
        @test val == [9.0, 4.0]
        @test jac == [6.0 0.0; 0.0 1.0]
    end

    @testset "@params with no fields" begin
        @eval @params struct EmptyParamsTest end
        @test EmptyParamsTest() isa EmptyParamsTest
    end

    @testset "correctsolution! fallback" begin
        m = Model(x -> x[1]^2)
        addvar!(m, [1.0], [2.0])
        vm, x0, _ = tovecmodel(m)
        g = getineqconstraints(vm)(x0)
        convstate = ConvergenceState{Float64}()
        s = Solution(x0 .+ 1, x0, 1.0, 2.0, 3.0, nothing, g, nothing, convstate)
        @test correctsolution!(s, vm, nothing) === s
    end
end
end
