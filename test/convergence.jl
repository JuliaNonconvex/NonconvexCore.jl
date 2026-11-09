@testitem "Convergence" begin
using Test, NonconvexCore
using NonconvexCore:
    Solution,
    ConvergenceState,
    assess_convergence!,
    hasconverged,
    Tolerance,
    GenericCriteria,
    KKTCriteria,
    ScaledKKTCriteria,
    IpoptCriteria

# A simple model min x^2 s.t. x >= 0 with optimum at x = 0.
m = Model(x -> x[1]^2)
addvar!(m, [-1.0], [1.0])
add_ineq_constraint!(m, x -> -x[1])
vm, _, _ = NonconvexCore.tovecmodel(m)

function solution_at(x, prevx, f, prevf, g)
    convstate = ConvergenceState{Float64}()
    return Solution(prevx, x, [0.0], prevf, f, [2 * x[1]], g, reshape([-1.0], 1, 1), convstate)
end

@testset "Tolerance helpers" begin
    tol = Tolerance()
    @test tol.x == 0.0
    @test tol.fabs == tol.frel == 1e-5
    @test tol.kkt == 1e-5
    @test tol.infeas == 1e-5
    tol2 = tol * 2
    @test tol2.fabs == 2e-5
    fadtol = Tolerance(x = i -> 1e-3 * i, f = i -> 1e-4 * i, kkt = i -> 1e-5 * i)
    tol3 = fadtol(5)
    @test tol3.x == 5e-3
    @test tol3.fabs == 5e-4
    @test tol3.kkt == 5e-5
end

@testset "Convergence criteria" begin
    tol = Tolerance()
    # A solution at the optimum: all criteria should report convergence.
    s = solution_at([0.0], [0.0], 0.0, 0.0, [0.0])
    for (criteria, field) in (
        (GenericCriteria(), :fabs_converged),
        (KKTCriteria(), :kkt_converged),
        (ScaledKKTCriteria(), :kkt_converged),
        (IpoptCriteria(), :ipopt_converged),
    )
        assess_convergence!(s, vm, tol, criteria, false, 1)
        @test s.convstate.converged
        @test s.convstate.infeas_converged
        @test getfield(s.convstate, field)
        @test hasconverged(s)
    end

    # An infeasible solution with a non-zero KKT residual: none converge.
    s2 = solution_at([-0.5], [-0.5], 0.25, 0.25, [0.5])
    for criteria in (GenericCriteria(), KKTCriteria(), ScaledKKTCriteria(), IpoptCriteria())
        assess_convergence!(s2, vm, tol, criteria, false, 1)
        @test !s2.convstate.converged
        @test !s2.convstate.infeas_converged
    end

    # The objective increased relative to the previous iterate.
    s3 = solution_at([0.4], [0.0], 0.16, 0.0, [-0.4])
    assess_convergence!(s3, vm, tol, GenericCriteria(), false, 1)
    @test s3.convstate.f_increased
end
end
