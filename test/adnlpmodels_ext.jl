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

@testitem "Exact JuMP evaluator backends" begin
    using NonconvexCore, ADNLPModels, NLPModels, Test, LinearAlgebra
    using JuMP: JuMP

    ext = Base.get_extension(NonconvexCore, :NonconvexCoreADNLPModelsExt)
    JuMPJacBackend = ext.JuMPJacBackend
    JuMPHvBackend = ext.JuMPHvBackend
    JuMPHessBackend = ext.JuMPHessBackend
    ZygoteJacobian = ext.ZygoteJacobian
    ZygoteHessian = ext.ZygoteHessian
    ZygoteHvprod = ext.ZygoteHvprod

    # A JuMP-converted model with a nonlinear objective and one nonlinear
    # inequality; the shared JuMPEvaluator drives the exact backends.
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:2] >= 0.5, start=1.0)
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + x[1] * x[2] + x[2]^2)
    JuMP.@constraint(jm, x[1]^2 + x[2] - 2 <= 0)
    dm = NonconvexCore.DictModel(jm)
    m = NonconvexCore.getobjective(dm).f.m   # the JuMPEvaluator
    vm, x0, _ = NonconvexCore.tovecmodel(dm) # vector-level model
    nvar = length(x0)
    ncon = 1
    f = NonconvexCore.getfunction(vm.objective)
    gfun = NonconvexCore.getfunction(vm.ineq_constraints.fs[1])
    c(y) = [gfun(y)]
    c! = (cx, y) -> (cx.=c(y); cx)

    meta = ADNLPModels.NLPModelMeta{Float64,Vector{Float64}}(
        nvar;
        x0=x0,
        lvar=fill(0.5, nvar),
        uvar=fill(Inf, nvar),
        ncon=ncon,
        y0=zeros(ncon),
        lcon=fill(-Inf, ncon),
        ucon=zeros(ncon),
        nnzj=nvar * ncon,
        nnzh=div(nvar * (nvar + 1), 2),
        lin=Int[],
        minimize=true,
        islp=false,
        name="exact-jump",
    )
    backend = ADNLPModels.ADModelBackend(
        nvar,
        f,
        ncon,
        c!;
        x0=x0,
        jump_evaluator=m,
        gradient_backend=ext.ZygoteGradient,
        hprod_backend=JuMPHvBackend,
        jprod_backend=ext.ZygoteJprod,
        jtprod_backend=ext.ZygoteJtprod,
        jacobian_backend=JuMPJacBackend,
        hessian_backend=JuMPHessBackend,
        ghjvprod_backend=ext.ZygoteGHjvprod,
    )
    nlp = ADNLPModels.ADNLPModel(meta, ADNLPModels.Counters(), backend, f, c!)

    # Exact Jacobian from the evaluator: gradient of x1^2 + x2 - 2
    J = Matrix(NLPModels.jac(nlp, x0))
    @test J ≈ [2x0[1] 1.0]

    # Exact Lagrangian Hessian (objective factor 1, multiplier 1):
    # H_f = [2 1; 1 2] (x1*x2 contributes the off-diagonal 1), H_g = [2 0; 0 0]
    y = [1.0]
    H = Matrix(NLPModels.hess(nlp, x0, y; obj_weight=1.0))
    @test H ≈ [4.0 1.0; 1.0 2.0]

    # Exact Lagrangian Hessian-vector products
    v = [1.0, -2.0]
    Hv = NLPModels.hprod(nlp, x0, y, v; obj_weight=1.0)
    @test Hv ≈ [4.0 1.0; 1.0 2.0] * v
    Hv2 = NLPModels.hprod(nlp, x0, [1.0], v; obj_weight=0.0)  # constraint 1 only
    @test Hv2 ≈ [2.0 0.0; 0.0 0.0] * v

    # The Zygote* constructors dispatch to the exact backends when a
    # JuMPEvaluator is passed, and stay Zygote-backed without one.
    @test ZygoteJacobian(nvar, f, ncon, c!; jump_evaluator=m) isa JuMPJacBackend
    @test ZygoteHessian(nvar, f, ncon, c!; jump_evaluator=m) isa JuMPHessBackend
    @test ZygoteHvprod(nvar, f, ncon, c!; jump_evaluator=m) isa JuMPHvBackend
    @test ZygoteHvprod(nvar, f, ncon, c!) isa ZygoteHvprod

    # The exact backends require a JuMPEvaluator.
    @test_throws ArgumentError JuMPJacBackend(nvar, f, ncon)
    @test_throws ArgumentError JuMPHvBackend(nvar, f, ncon)
    @test_throws ArgumentError JuMPHessBackend(nvar, f, ncon)
end
