using NonconvexCore, Test
using NonconvexCore: AbstractOptimizer, GenericResult, VecModel
import NonconvexCore: Workspace

# Dummy solvers with different declaration levels. `SilentSolver` has no
# declaration; the others declare capabilities and have a trivial workspace so
# they can be exercised through the generic `optimize`.
struct DeclaredSolver <: AbstractOptimizer end
struct PartialSolver <: AbstractOptimizer end
struct LyingSolver <: AbstractOptimizer end
struct SilentSolver <: AbstractOptimizer end

NonconvexCore.solver_capabilities(::DeclaredSolver) =
    ConstraintCapabilities(
        linear_eq = true, nonlinear_eq = true,
        linear_ineq = true, nonlinear_ineq = true,
        linear_sd = true, nonlinear_sd = true,
        box = true, continuous = true, binary = true, integer = true,
        infinite_bounds = true,
    )
NonconvexCore.solver_capabilities(::PartialSolver) =
    ConstraintCapabilities(
        linear_eq = true, nonlinear_eq = false,
        linear_ineq = true, nonlinear_ineq = false,
        linear_sd = true, nonlinear_sd = false,
        box = true, continuous = true, binary = false, integer = false,
        infinite_bounds = false,
    )
NonconvexCore.solver_capabilities(::LyingSolver) =
    ConstraintCapabilities(linear_eq = true, linear_ineq = true, box = true)

struct ToyWorkspace <: Workspace
    model::VecModel
    x0::Vector{Float64}
end
for S in (DeclaredSolver, PartialSolver, LyingSolver, SilentSolver)
    @eval function Workspace(
        model::VecModel,
        ::$S,
        x0::AbstractVector;
        kwargs...,
    )
        return ToyWorkspace(model, collect(float.(x0)))
    end
end
function NonconvexCore.optimize!(w::ToyWorkspace)    x = copy(w.x0)
    f = w.model.objective(x)
    fv = f isa Real ? f : sum(f)
    return GenericResult(
        nothing,
        copy(w.x0),
        x,
        fv,
        1,
        false,
        Tolerance(),
        nothing,
        1,
    )
end

# Helper: capture the exception thrown by `f` (works across Julia versions
# regardless of what `@test_throws` returns).
function _capture(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

function make_model(; eq = false, neq = false, ineq = false, nineq = false,
                     sd = false, box = false, integer = false, binary = false)
    m = Model()
    if box
        addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    else
        addvar!(m, fill(-Inf, 2), fill(Inf, 2))
    end
    integer && setinteger!(m, 1, true)
    binary && begin
        setinteger!(m, 1, true)
        setmin!(m, 1, 0.0)
        setmax!(m, 1, 1.0)
    end
    eq && add_eq_constraint!(m, x -> sum(x) - 1)
    neq && add_eq_constraint!(m, x -> x[1]^2 - 1)
    ineq && add_ineq_constraint!(m, x -> x[1] - 0.5)
    nineq && add_ineq_constraint!(m, x -> x[1]^2 - 0.5)
    sd && add_sd_constraint!(m, x -> Matrix{Float64}(I, 2, 2))
    return m
end

@testset "ConstraintCapabilities" begin
    c = ConstraintCapabilities()
    @test c.linear_eq === missing
    @test c.nonlinear_eq === missing
    @test c.linear_ineq === missing
    @test c.nonlinear_ineq === missing
    @test c.linear_sd === missing
    @test c.nonlinear_sd === missing
    @test c.box === missing
    @test c.continuous === missing
    @test c.binary === missing
    @test c.integer === missing
    @test c.infinite_bounds === missing
    c2 = ConstraintCapabilities(linear_eq = true, nonlinear_sd = false)
    @test c2.linear_eq === true
    @test c2.nonlinear_sd === false
    @test c2.box === missing

    merged = NonconvexCore.merge_capabilities(c2; nonlinear_ineq = true, integer = true)
    @test merged.linear_eq === true
    @test merged.nonlinear_sd === false
    @test merged.nonlinear_ineq === true
    @test merged.integer === true
    @test merged.box === missing
    merged2 = NonconvexCore.merge_capabilities(nothing; linear_eq = true)
    @test merged2.linear_eq === true
    @test merged2.nonlinear_sd === missing
end

@testset "infinite bounds capability" begin
    # Declared-supported infinite bounds pass.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(), ones(2))
    @test checksubproblem!(m, DeclaredSolver()) === nothing

    # Declared-false errors with an actionable message.
    err = _capture() do
        checksubproblem!(m, PartialSolver())
    end
    @test err isa ArgumentError
    @test occursin("PartialSolver", sprint(showerror, err))
    @test occursin("infinite", sprint(showerror, err))

    # Undeclared errors strictly.
    err = _capture() do
        checksubproblem!(m, LyingSolver())
    end
    @test err isa ArgumentError
    @test occursin("solver_capabilities", sprint(showerror, err))

    # A model with finite bounds only does not trigger the check.
    m2, x0, _ = NonconvexCore.tovecmodel(make_model(; box = true), ones(2))
    @test checksubproblem!(m2, PartialSolver()) === nothing

    # The merge forwards the sub-solver's declaration.
    merged = NonconvexCore.merge_capabilities(
        NonconvexCore.solver_capabilities(PartialSolver()); nonlinear_ineq = true,
    )
    @test merged.infinite_bounds === false

    # optimize integration.
    err = _capture() do
        optimize(make_model(), PartialSolver(), ones(2))
    end
    @test err isa ArgumentError
    @test occursin("infinite", sprint(showerror, err))
end

@testset "checksubproblem! basics" begin
    # Fully declared solver passes on any model.
    for kwargs in (
        (;),
        (; eq = true),
        (; neq = true),
        (; box = true),
        (; integer = true),
        (; binary = true),
        (; eq = true, ineq = true, neq = true, nineq = true, sd = true, box = true, integer = true, binary = true),
    )
        m, x0, _ = NonconvexCore.tovecmodel(make_model(; kwargs...), ones(2))
        @test checksubproblem!(m, DeclaredSolver()) === nothing
    end

    # Feature-free models pass for any solver, declared or not.
    m, x0, _ = NonconvexCore.tovecmodel(Model(), Float64[])
    @test checksubproblem!(m, SilentSolver()) === nothing
    @test solver_capabilities(SilentSolver()) === nothing

    # Undeclared solvers error on models with any feature (strict by default).
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; eq = true, sd = true), ones(2))
    err = _capture() do
        checksubproblem!(m, SilentSolver())
    end
    @test err isa ArgumentError
    @test occursin("solver_capabilities", sprint(showerror, err))
end

@testset "checksubproblem! errors" begin
    # Declared-false features error with an actionable message.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; nineq = true), ones(2))
    err = _capture() do
        checksubproblem!(m, PartialSolver())
    end
    @test err isa ArgumentError
    @test occursin("PartialSolver", sprint(showerror, err))
    @test occursin("nonlinear inequality", sprint(showerror, err))

    m, x0, _ = NonconvexCore.tovecmodel(make_model(; integer = true), ones(2))
    @test _capture(() -> checksubproblem!(m, PartialSolver())) isa ArgumentError

    # Undeclared (missing) features error: declare or drop them.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; sd = true), ones(2))
    err = _capture() do
        checksubproblem!(m, LyingSolver())
    end
    @test err isa ArgumentError
    @test occursin("solver_capabilities", sprint(showerror, err))

    # Assumed capabilities override the solver's declaration.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; box = true, ineq = true), ones(2))
    @test checksubproblem!(
        m,
        PartialSolver();
        assumed = ConstraintCapabilities(; linear_ineq = true, box = true, continuous = true, infinite_bounds = true),
    ) === nothing

    # Hint text is appended to the error.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; nineq = true), ones(2))
    err = _capture() do
        checksubproblem!(
            m,
            PartialSolver();
            hint = "relax the constraint",
        )
    end
    @test err isa ArgumentError
    @test occursin("relax the constraint", sprint(showerror, err))
end

@testset "optimize integration" begin
    # Solvers with a declaration are validated by the generic optimize.
    m = make_model(; nineq = true)
    err = _capture() do
        optimize(m, PartialSolver(), ones(2))
    end
    @test err isa ArgumentError
    @test occursin("nonlinear inequality", sprint(showerror, err))

    # Declared-and-supported features pass and produce a result.
    m = make_model(; box = true)
    r = optimize(m, DeclaredSolver(), ones(2))
    @test r isa GenericResult

    # Undeclared solvers error when the model has sd constraints, instead of
    # the old warning that silently dropped them.
    m = make_model(; sd = true)
    err = _capture() do
        optimize(m, SilentSolver(), ones(2))
    end
    @test err isa ArgumentError
    @test occursin("solver_capabilities", sprint(showerror, err))
end
