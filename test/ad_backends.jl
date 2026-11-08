using NonconvexCore, LinearAlgebra, Test, Zygote, ForwardDiff, SparseArrays
import ADTypes
import DifferentiationInterface as DI

const f = x -> sum(abs2, x)
const g = x -> [x[1] + x[2], x[3]^2, x[4] * x[5]]
const x = [0.3, -0.7, 1.2, 0.5, -0.4]

@testset "AD backends - value and derivatives" begin
    v0, g0 = NonconvexCore.value_gradient(f, x)
    @test v0 == f(x)
    @test g0 ≈ Zygote.gradient(f, x)[1]
    vf, gf = NonconvexCore.value_gradient(f, x, ADTypes.AutoForwardDiff())
    @test vf == v0
    @test gf ≈ g0

    val, jac = NonconvexCore.value_jacobian(f, x)
    @test val == f(x)
    @test jac isa Adjoint
    @test jac ≈ Zygote.jacobian(f, x)[1]
    valf, jacf = NonconvexCore.value_jacobian(f, x, ADTypes.AutoForwardDiff())
    @test valf == val
    @test jacf ≈ jac

    m = Model(f)
    addvar!(m, -ones(5), ones(5))
    add_ineq_constraint!(m, g)
    vm, xv, _ = NonconvexCore.tovecmodel(m)
    c = vm.ineq_constraints
    vc, jc = NonconvexCore.value_jacobian(c, xv)
    @test vc == g(xv)
    @test jc isa Adjoint
    @test jc ≈ Zygote.jacobian(g, xv)[1]
    vcf, jcf = NonconvexCore.value_jacobian(c, xv, ADTypes.AutoForwardDiff())
    @test vcf ≈ vc
    @test jcf ≈ jc

    empty_c = NonconvexCore.VectorOfFunctions(())
    ve, je = NonconvexCore.value_jacobian(empty_c, xv)
    @test isempty(ve)
    @test size(je) == (0, 5)

    vt, gt = NonconvexCore.value_jacobian_transpose(f, x)
    @test vt == f(x)
    @test gt ≈ g0
end

@testset "AD backends - sparse derivatives" begin
    m = Model(f)
    addvar!(m, -ones(5), ones(5))
    add_ineq_constraint!(m, g)
    vm, xv, _ = NonconvexCore.tovecmodel(m)
    c = vm.ineq_constraints

    J = NonconvexCore.sparse_jacobian(c, xv)
    @test J isa SparseMatrixCSC
    @test J ≈ Zygote.jacobian(g, xv)[1]
    Jf = NonconvexCore.sparse_jacobian(c, xv, ADTypes.AutoForwardDiff())
    @test Jf ≈ J

    Hs = NonconvexCore.sparse_hessian(f, xv)
    @test Hs isa SparseMatrixCSC
    @test Hs ≈ Zygote.hessian(f, xv)
end

@testset "AD backends - multi-constraint collections" begin
    m = Model(f)
    addvar!(m, -ones(5), ones(5))
    add_ineq_constraint!(m, x -> x[1] + x[2])
    add_ineq_constraint!(m, x -> x[3]^2 - x[4])
    add_eq_constraint!(m, x -> x[2] * x[5])
    vm, xv, _ = NonconvexCore.tovecmodel(m)
    c = vm.ineq_constraints
    ceq = vm.eq_constraints

    # Reverse-mode differentiation of the whole collection must give the
    # per-constraint derivatives (Zygote's vcat pullback used to corrupt
    # seeds for collections with more than one function).
    Jz = Matrix(Zygote.jacobian(c, xv)[1])
    Jf = Matrix(NonconvexCore.value_jacobian(c, xv, ADTypes.AutoForwardDiff())[2])
    @test Jz ≈ Jf
    Jz_eq = Matrix(Zygote.jacobian(ceq, xv)[1])
    Jf_eq = Matrix(NonconvexCore.value_jacobian(ceq, xv, ADTypes.AutoForwardDiff())[2])
    @test Jz_eq ≈ Jf_eq

    # The sparse pipeline gives the same values.
    Js = NonconvexCore.sparse_jacobian(c, xv)
    @test Js ≈ Jf
    Js_eq = NonconvexCore.sparse_jacobian(ceq, xv, ADTypes.AutoForwardDiff())
    @test Js_eq ≈ Jf_eq

    # Prepared evaluation reuses the detected pattern across calls: repeated
    # evaluations at other points keep the same structure and correct values.
    xv2 = xv + 0.1 .* sin.(1:length(xv))
    Js2 = NonconvexCore.sparse_jacobian(c, xv2)
    @test nnz(Js2) == nnz(Js)
    Jf2 = Matrix(NonconvexCore.value_jacobian(c, xv2, ADTypes.AutoForwardDiff())[2])
    @test Js2 ≈ Jf2

    # The Hessian preparation is cached by function type, so a fresh closure
    # instance of the same type reuses it.
    Hs1 = NonconvexCore.sparse_hessian(f, xv)
    Hs2 = NonconvexCore.sparse_hessian(f, xv2)
    @test nnz(Hs1) == nnz(Hs2)
    @test Hs2 ≈ Zygote.hessian(f, xv2)
end

@testset "AD backends - default backend" begin
    old = get_ad_backend()
    set_ad_backend!(ADTypes.AutoForwardDiff())
    @test get_ad_backend() isa ADTypes.AutoForwardDiff
    v, grad = NonconvexCore.value_gradient(f, x)
    @test grad ≈ Zygote.gradient(f, x)[1]
    set_ad_backend!(old)
    @test get_ad_backend() === old
end

@testset "AD backends - Zygote sparse path" begin
    # Regression: the Zygote branch of sparse_jacobian used to read its
    # sparsity pattern outside the scope it was assigned in.
    g = z -> [z[1] + z[2], z[1]^2, 3z[2]]
    J = Matrix(NonconvexCore.sparse_jacobian(g, [1.0, 2.0], ADTypes.AutoZygote()))
    @test J ≈ [1.0 1.0; 2.0 0.0; 0.0 3.0]
end
