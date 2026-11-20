@testitem "Custom gradients" begin
    using NonconvexCore, Test, Zygote, ForwardDiff, SparseArrays, Tracker, LinearAlgebra
    using ChainRulesCore
    using ChainRulesCore

    @testset "CustomGradFunction" begin
        fakeg = [2.0]
        f = CustomGradFunction(sum, x -> fakeg)
        @test Zygote.gradient(f, [1.0]) == (fakeg,)
        @test ForwardDiff.gradient(f, [1.0]) == fakeg

        fakeJ = [1.0 2.0; 0.0 -1.0]
        f = CustomGradFunction(identity, x -> fakeJ)
        @test Zygote.jacobian(f, [1.0, 1.0]) == (fakeJ,)
        @test ForwardDiff.jacobian(f, [1.0, 1.0]) == fakeJ
    end

    @testset "CustomHessianFunction" begin
        fakeH = [3.0 -1.0; -1.0 2.0]
        f = CustomHessianFunction(sum, x -> fakeg, x -> fakeH)
        fakeg = [2.0, 2.0]
        @test Zygote.gradient(f, [1.0, 1.0]) == (fakeg,)
        @test Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0]) == (fakeH,)
        @test ForwardDiff.gradient(f, [1.0, 1.0]) == fakeg
        @test ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0]) == fakeH
        @test ForwardDiff.jacobian(x -> ForwardDiff.gradient(f, x), [1.0, 1.0]) == fakeH

        hvp = (x, v) -> fakeH * v
        f = CustomHessianFunction(sum, x -> fakeg, hvp; hvp=true)
        H = Zygote.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0])[1]
        @test norm(H - fakeH) < 1e-6
        H = ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], [1.0, 1.0])
        @test norm(H - fakeH) < 1e-6
        H = ForwardDiff.jacobian(x -> ForwardDiff.gradient(f, x), [1.0, 1.0])
        @test norm(H - fakeH) < 1e-6
    end

    @testset "CustomGradFunction rrule with InplaceableThunk" begin
        fakeJ = [1.0 2.0; 0.0 -1.0]
        f = CustomGradFunction(identity, x -> fakeJ)
        v, pullback = ChainRulesCore.rrule(f, [1.0, 1.0])
        Δ = ChainRulesCore.InplaceableThunk(
            (d, x) -> d, ChainRulesCore.Thunk(() -> [1.0, 1.0])
        )
        result = pullback(Δ)
        @test result[2] == fakeJ' * [1.0, 1.0]
    end

    @testset "CustomHessianFunction rrule with InplaceableThunk" begin
        fakeg = [2.0, 2.0]
        fakeH = [3.0 -1.0; -1.0 2.0]
        f = CustomHessianFunction(sum, x -> fakeg, x -> fakeH)
        v, pullback = ChainRulesCore.rrule(f, [1.0, 1.0])
        Δ = ChainRulesCore.InplaceableThunk((d, x) -> d, ChainRulesCore.Thunk(() -> 1.0))
        result = pullback(Δ)
        @test result[2] == fakeg * 1.0
    end

    @testset "CustomGradFunction evaluates the custom gradient once per forward pass" begin
        # Regression test: the custom gradient used to be evaluated on every
        # pullback application (once per Jacobian row in value_jacobian).
        calls = Ref(0)
        g = x -> (calls[] += 1; [1.0 0.0; 0.0 1.0])
        f = CustomGradFunction(identity, g)
        calls[] = 0
        val, jac = NonconvexCore.value_jacobian(f, [1.0, 2.0])
        @test val == [1.0, 2.0]
        @test jac == [1.0 0.0; 0.0 1.0]
        @test calls[] == 1
    end
end

@testitem "Vector-valued custom Hessian via HVP" begin
    using NonconvexCore, Test, Zygote, ForwardDiff, LinearAlgebra

    # a vector-valued constraint c: R^2 -> R^2 with custom Jacobian and a
    # per-component HVP operator (issue #67)
    c(x) = [x[1]^2 + x[2], x[1] * x[2]]
    J(x) = [2x[1] 1.0; x[2] x[1]]
    # ∇²c₁ = [[2, 0], [0, 0]], ∇²c₂ = [[0, 1], [1, 0]]
    hvp(x, v) = [2v[1] 0.0; v[2] v[1]]

    f = CustomHessianFunction(c, J, hvp; hvp=true)
    x0 = [1.5, 2.5]
    μ = [3.0, 4.0]

    # value and Jacobian
    @test f(x0) == c(x0)
    @test f.g(x0) == J(x0)

    # gradient of the weighted Lagrangian μ'c
    g = Zygote.gradient(x -> dot(μ, f(x)), x0)[1]
    @test g ≈ J(x0)'μ

    # Hessian of the weighted Lagrangian assembled from the HVPs:
    # Σ μᵢ∇²cᵢ = [[2μ₁, μ₂], [μ₂, 0]]
    H = Zygote.hessian(x -> dot(μ, f(x)), x0)
    @test H ≈ [2μ[1] μ[2]; μ[2] 0.0]

    # ForwardDiff agrees
    Hfd = ForwardDiff.hessian(x -> dot(μ, f(x)), x0)
    @test Hfd ≈ H

    # scalar-valued f with hvp = true still works
    s(x) = x[1]^2 + 3x[2]^2 + x[1] * x[2]
    hs = CustomHessianFunction(
        s,
        x -> [2x[1] + x[2], 6x[2] + x[1]],
        (x, v) -> [2v[1] + v[2], v[1] + 6v[2]];
        hvp=true,
    )
    @test Zygote.hessian(hs, [1.0, 2.0]) ≈ [2.0 1.0; 1.0 6.0]
end
