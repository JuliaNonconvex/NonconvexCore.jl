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
    ConstraintCapabilities(eq = true, ineq = true, sd = true, box = true, integer = true)
NonconvexCore.solver_capabilities(::PartialSolver) =
    ConstraintCapabilities(eq = true, ineq = false, sd = true, box = true, integer = false)
NonconvexCore.solver_capabilities(::LyingSolver) =
    ConstraintCapabilities(eq = true, ineq = true, box = true)

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

function make_model(; eq = false, ineq = false, sd = false, box = false, integer = false)
    m = Model()
    if box
        addvar!(m, [-1.0, -1.0], [1.0, 1.0])
    else
        addvar!(m, fill(-Inf, 2), fill(Inf, 2))
    end
    integer && setinteger!(m, 1, true)
    eq && add_eq_constraint!(m, x -> sum(x) - 1)
    ineq && add_ineq_constraint!(m, x -> x[1] - 0.5)
    sd && add_sd_constraint!(m, x -> Matrix{Float64}(I, 2, 2))
    return m
end

@testset "ConstraintCapabilities" begin
    c = ConstraintCapabilities()
    @test c.eq === missing
    @test c.ineq === missing
    @test c.sd === missing
    @test c.box === missing
    @test c.integer === missing
    c2 = ConstraintCapabilities(eq = true, sd = false)
    @test c2.eq === true
    @test c2.sd === false
    @test c2.box === missing

    merged = NonconvexCore.merge_capabilities(c2; ineq = true, integer = true)
    @test merged.eq === true
    @test merged.sd === false
    @test merged.ineq === true
    @test merged.integer === true
    @test merged.box === missing
    merged2 = NonconvexCore.merge_capabilities(nothing; eq = true)
    @test merged2.eq === true
    @test merged2.sd === missing
end

@testset "checksubproblem! basics" begin
    # Fully declared solver passes on any model.
    for kwargs in (
        (;),
        (; eq = true),
        (; box = true),
        (; integer = true),
        (; eq = true, ineq = true, sd = true, box = true, integer = true),
    )
        m, x0, _ = NonconvexCore.tovecmodel(make_model(; kwargs...), ones(2))
        @test checksubproblem!(m, DeclaredSolver()) === nothing
    end

    # Feature-free models pass for any solver, declared or not.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(), ones(2))
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
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; ineq = true), ones(2))
    err = _capture() do
        checksubproblem!(m, PartialSolver())
    end
    @test err isa ArgumentError
    @test occursin("PartialSolver", sprint(showerror, err))
    @test occursin("ineq", sprint(showerror, err))

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
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; ineq = true), ones(2))
    @test checksubproblem!(
        m,
        PartialSolver();
        assumed = ConstraintCapabilities(; ineq = true),
    ) === nothing

    # Hint text is appended to the error.
    m, x0, _ = NonconvexCore.tovecmodel(make_model(; ineq = true), ones(2))
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
    m = make_model(; ineq = true)
    err = _capture() do
        optimize(m, PartialSolver(), ones(2))
    end
    @test err isa ArgumentError
    @test occursin("ineq", sprint(showerror, err))

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
