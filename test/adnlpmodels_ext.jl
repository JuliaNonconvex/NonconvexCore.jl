@testitem "ADNLPModels compat backends" begin
    using NonconvexCore, ADNLPModels, Test, LinearAlgebra

    # Loading ADNLPModels triggers NonconvexCoreADNLPModelsExt, which defines
    # the Zygote*-named ADNLPModels backends on top of NonconvexCore's AD.
    ext = Base.get_extension(NonconvexCore, :NonconvexCoreADNLPModelsExt)
    @test ext !== nothing
    @test isdefined(ext, :ZygoteGradient)
    @test isdefined(ext, :ZygoteHessian)

    ZygoteGradient = ext.ZygoteGradient
    ZygoteJacobian = ext.ZygoteJacobian
    ZygoteJprod = ext.ZygoteJprod
    ZygoteJtprod = ext.ZygoteJtprod
    ZygoteHessian = ext.ZygoteHessian
    ZygoteHvprod = ext.ZygoteHvprod
    ZygoteGHjvprod = ext.ZygoteGHjvprod

    f(x) = x[1]^2 + 3.0 * x[1] * x[2] + 2.0 * x[2]^2 + x[1] - x[2]
    x = [1.0, -2.0]
    ∇f = [2x[1] + 3x[2] + 1, 3x[1] + 4x[2] - 1]
    Hf = [2.0 3.0; 3.0 4.0]
    c(y) = [y[1] + y[2], y[1] * y[2]]
    Jc = [1.0 1.0; x[2] x[1]]

    b_g = ZygoteGradient(length(x), f)
    @test ADNLPModels.gradient(b_g, f, x) ≈ ∇f atol = 1e-12
    g = zeros(2)
    @test ADNLPModels.gradient!(b_g, g, f, x) ≈ ∇f atol = 1e-12

    b_J = ZygoteJacobian(length(x), f, 2, (args...) -> [])
    @test ADNLPModels.jacobian(b_J, c, x) ≈ Jc atol = 1e-12

    Jv = zeros(2)
    @test ADNLPModels.Jprod!(ZygoteJprod(), Jv, c, x, [2.0, -1.0], Val(1)) ≈
        Jc * [2.0, -1.0] atol = 1e-12
    Jtv = zeros(2)
    @test ADNLPModels.Jtprod!(ZygoteJtprod(), Jtv, c, x, [1.0, -1.0], Val(1)) ≈
        Jc' * [1.0, -1.0] atol = 1e-12

    b_H = ZygoteHessian(length(x), f)
    @test ADNLPModels.hessian(b_H, f, x) ≈ Hf atol = 1e-12

    Hv = zeros(2)
    v = [1.0, 2.0]
    @test ADNLPModels.Hvprod!(ZygoteHvprod(), Hv, x, v, f, Val(:obj), 1.0) ≈ Hf * v atol =
        1e-12
    @test ADNLPModels.Hvprod!(ZygoteHvprod(), Hv, x, v, f, Val(:lag), [1.0], 2.0) ≈ Hf * v atol =
        1e-12
    @test ADNLPModels.Hvprod!(ZygoteHvprod(), Hv, x, v, f, Val(:ci)) ≈ Hf * v atol = 1e-12

    w = [1.0, -1.0]
    @test ADNLPModels.directional_second_derivative(ZygoteGHjvprod(), f, x, v, w) ≈
        dot(w, Hf * v) atol = 1e-12

    # Backend constructors match ADNLPModels' calling convention.
    @test ZygoteGradient(length(x), f) isa ZygoteGradient
    @test ZygoteJacobian(length(x), f, 2).nnzj == 2 * length(x)
    @test ZygoteJprod() isa ZygoteJprod
    @test ZygoteJtprod() isa ZygoteJtprod
    @test ZygoteHessian(length(x), f).nnzh == div(length(x) * (length(x) + 1), 2)
    @test ZygoteHvprod() isa ZygoteHvprod
    @test ZygoteGHjvprod() isa ZygoteGHjvprod
end
