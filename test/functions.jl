using NonconvexCore
using NonconvexCore: VectorOfFunctions, FunctionWrapper, CountingFunction, 
                     getdim, getfunction, getfunctions, Objective, IneqConstraint,
                     EqConstraint, SDConstraint
using Test

@testset "CountingFunction" begin
    f = CountingFunction(FunctionWrapper(log, 1))
    for i in 1:3
        x = rand()
        @test log(x) == f(x)
        @test f.counter[] == i
    end
    @test getdim(f) == 1
    @test getfunction(f) isa FunctionWrapper
    @test getfunction(getfunction(f)) === log
    @test length(f) == getdim(f)
end

@testset "VectorOfFunctions" begin
    f1 = FunctionWrapper(log, 1)
    f2 = FunctionWrapper(exp, 1)
    fs = [f1, f2]
    f = VectorOfFunctions([f1, f2])
    f == VectorOfFunctions(Any[f])
    for _ in 1:3
        x = rand()
        @test [log(x), exp(x)] == f(x)
    end
    @test getdim(f) == 2
    @test getfunctions(f) == fs
    @test getfunction(f, 1) == fs[1]
    @test getfunction(f, 2) == fs[2]
end

@testset "Objective" begin
    f = Objective(log)
    @test getdim(f) == 1
    for _ in 1:3
        x = rand()
        @test f(x) == log(x)
    end
    @test getfunction(f) === log
end

@testset "IneqConstraint" begin
    f = IneqConstraint(log, 1.0)
    for _ in 1:3
        x = rand()
        @test f(x) == log(x) - 1
    end
    @test getdim(f) == 1
    @test getfunction(f) === log
end

@testset "EqConstraint" begin
    f = EqConstraint(log, 1.0)
    for _ in 1:3
        x = rand()
        @test f(x) == log(x) - 1
    end
    @test getdim(f) == 1
    @test getfunction(f) === log
end

@testset "SDConstraint" begin
    f = SDConstraint(exp, 1)
    for _ in 1:3
        x = rand()
        @test f(x) == exp(x)
    end
    @test getdim(f) == 1
    @test getfunction(f) === exp

    f = SDConstraint(exp, 2)
    for _ in 1:3
        x = rand(2, 2)
        @test f(x) == exp(x)
    end
    @test getdim(f) == 2
    @test getfunction(f) === exp
end
