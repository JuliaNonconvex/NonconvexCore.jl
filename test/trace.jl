@testitem "TraceFunction" begin
    using NonconvexCore, Test, Zygote, ForwardDiff, LinearAlgebra
    @testset "TraceFunction" begin
        @testset "Reverse-mode" begin
            f = TraceFunction(sum, on_call=true)
            @test f.on_call == true
            @test f.on_grad == false
            f([2.0])
            @test f.trace == [(input=[2.0], output=2.0)]
            Zygote.gradient(f, [3.0])
            @test f.trace == [(input=[2.0], output=2.0)]

            f = TraceFunction(sum, on_grad=true)
            @test f.on_call == false
            @test f.on_grad == true
            f([2.0])
            @test f.trace == []
            Zygote.gradient(f, [3.0])
            @test f.trace == [(input=[3.0], output=3.0, grad=[[1.0]])]

            f = TraceFunction(sum)
            @test f.on_call == true
            @test f.on_grad == true
            f = TraceFunction(sum, on_call=true, on_grad=true)
            @test f.on_call == true
            @test f.on_grad == true
            f([2.0])
            @test f.trace == [(input=[2.0], output=2.0)]
            Zygote.gradient(f, [3.0])
            @test f.trace ==
                [(input=[2.0], output=2.0), (input=[3.0], output=3.0, grad=[[1.0]])]

            f = TraceFunction(sum, on_call=false, on_grad=false)
            @test f.on_call == false
            @test f.on_grad == false
            f([2.0])
            @test f.trace == []
            Zygote.gradient(f, [3.0])
            @test f.trace == []

            # Scalar input in reverse mode previously crashed with
            # FieldError: type Float64 has no field val.
            tf = TraceFunction(sin, on_grad=true)
            @test Zygote.gradient(tf, 1.0)[1] ≈ cos(1.0)
            @test length(tf.trace) == 1

            # Primal call followed by AD at the same point previously crashed with
            # FieldError: type NamedTuple has no field grad (default config).
            tf2 = TraceFunction(sum)
            tf2([2.0, 2.0])
            ForwardDiff.gradient(tf2, [3.0, 3.0])
            @test length(tf2.trace) == 2

            # Regression: the rrule stores a LIST of per-pass gradients (the
            # same format the frules append to). The rrule previously stored
            # the raw gradient, so an frule firing later at the same point
            # would push a vector gradient into a raw matrix (MethodError).
            tf3 = TraceFunction(x -> [sin(x[1]), cos(x[2])], on_grad=true)
            Zygote.jacobian(tf3, [1.0, 2.0])
            @test !isempty(tf3.trace)
            @test all(e -> e.grad isa AbstractVector, tf3.trace)
            @test all(e -> all(g -> g isa AbstractArray, e.grad), tf3.trace)
        end

        @testset "Forward-mode" begin
            f = TraceFunction(sum, on_call=true)
            @test f.on_call == true
            @test f.on_grad == false
            f([2.0, 2.0])
            @test f.trace == [(input=[2.0, 2.0], output=4.0)]
            ForwardDiff.gradient(f, [3.0, 3.0])
            @test f.trace == [(input=[2.0, 2.0], output=4.0)]

            f = TraceFunction(sum, on_grad=true)
            @test f.on_call == false
            @test f.on_grad == true
            f([2.0, 2.0])
            @test f.trace == []
            ForwardDiff.gradient(f, [3.0, 3.0])
            @test f.trace == [(input=[3.0, 3.0], output=6.0, grad=[1.0, 1.0])]

            f = TraceFunction(sum)
            @test f.on_call == true
            @test f.on_grad == true
            f = TraceFunction(sum, on_call=true, on_grad=true)
            @test f.on_call == true
            @test f.on_grad == true
            f([2.0, 2.0])
            @test f.trace == [(input=[2.0, 2.0], output=4.0)]
            ForwardDiff.gradient(f, [3.0, 3.0])
            @test f.trace == [
                (input=[2.0, 2.0], output=4.0),
                (input=[3.0, 3.0], output=6.0, grad=[1.0, 1.0]),
            ]

            f = TraceFunction(sum, on_call=false, on_grad=false)
            @test f.on_call == false
            @test f.on_grad == false
            f([2.0, 2.0])
            @test f.trace == []
            ForwardDiff.gradient(f, [3.0, 3.0])
            @test f.trace == []
        end
    end
end
