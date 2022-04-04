struct JuMPProblem{E, M, V}
    evaluator::E
    model::M
    vars::V
end

struct JuMPEvaluator{XL, XU, X, CL, CU, O, C, G, J, JS, H, HS} <: MOI.AbstractNLPEvaluator
    nvars::Int
    xlb::XL
    xub::XU
    x0::X
    nconstr::Int
    clb::CL
    cub::CU
    njacvalues::Int
    nhessvalues::Int
    obj::O
    eval_g::C
    eval_grad_f::G
    eval_jac_g::J
    jac_structure::JS
    eval_h::H
    lag_hess_structure::HS
end

function get_jump_problem(
    model::VecModel, x0 = getinit(model);
    optimizer, first_order, kwargs...,
)
    integers = model.integer
    eq = if length(model.eq_constraints.fs) == 0
        nothing
    else
        model.eq_constraints
    end
    ineq = if length(model.ineq_constraints.fs) == 0
        nothing
    else
        model.ineq_constraints
    end
    obj = CountingFunction(getobjective(model))
    return get_jump_problem(
        obj, ineq, eq, x0, integers, getmin(model),
        getmax(model), first_order, optimizer,
    ), obj.counter
end
function get_jump_problem(
    obj, ineq_constr, eq_constr, x0, integers,
    xlb, xub, first_order, optimizer,
)
    nvars = 0
    if ineq_constr !== nothing
        ineqJ0 = Zygote.jacobian(ineq_constr, x0)[1]
        ineq_nconstr, nvars = size(ineqJ0)
        Joffset = nvalues(ineqJ0)
    else
        ineqJ0 = nothing
        ineq_nconstr = 0
        Joffset = 0
    end
    if eq_constr !== nothing
        eqJ0 = Zygote.jacobian(eq_constr, x0)[1]
        eq_nconstr, nvars = size(eqJ0)
    else
        eqJ0 = nothing
        eq_nconstr = 0
    end
    njacvals = nvalues(ineqJ0) + nvalues(eqJ0)
    @assert nvars > 0
    lag(factor, y) = x -> begin
        factor * obj(x) + 
            _dot(ineq_constr, x, @view(y[1:ineq_nconstr])) + 
            _dot(eq_constr, x, @view(y[ineq_nconstr+1:end]))
    end
    clb = [fill(-Inf, ineq_nconstr); zeros(eq_nconstr)]
    cub = zeros(ineq_nconstr + eq_nconstr)

    function eval_g(x::AbstractVector{Float64}, g::AbstractVector{Float64})
        if ineq_constr !== nothing
            g[1:ineq_nconstr] .= ineq_constr(x)
        end
        if eq_constr !== nothing
            g[ineq_nconstr+1:end] .= eq_constr(x)
        end
        return g
    end
    function eval_grad_f(x::AbstractVector{Float64}, grad_f::AbstractVector{Float64})
        grad_f .= Zygote.gradient(obj, x)[1]
    end
    function jac_structure()
        rows = fill(0, njacvals)
        cols = fill(0, njacvals)
        ineqJ0 === nothing || fill_indices!(rows, cols, ineqJ0)
        eqJ0 === nothing || fill_indices!(rows, cols, eqJ0, offset = Joffset, row_offset = ineq_nconstr)
        return tuple.(rows, cols)
    end
    function eval_jac_g(x::AbstractVector{Float64}, values::AbstractVector{Float64})
        values .= 0
        if ineq_constr !== nothing
            ineqJ = Zygote.jacobian(ineq_constr, x)[1]
            add_values!(values, ineqJ)
        end
        if eq_constr !== nothing
            eqJ = Zygote.jacobian(eq_constr, x)[1]
            add_values!(values, eqJ, offset = Joffset)
        end
        return values
    end

    if first_order
        eval_h = (x...) -> 0.0
        lag_hess_structure = () -> Tuple{Int,Int}[]
        Hnvalues = 0
    else
        HL0 = LowerTriangular(
            Zygote.hessian(
                lag(1.0, ones(ineq_nconstr + eq_nconstr)),
                x0,
            ),
        )
        Hnvalues = nvalues(HL0)
        lag_hess_structure = function ()
            rows = fill(0, Hnvalues)
            cols = fill(0, Hnvalues)
            fill_indices!(rows, cols, HL0)
            return tuple.(rows, cols)
        end
        eval_h = function (x::AbstractVector{Float64}, obj_factor::Float64, lambda::AbstractVector{Float64}, values::AbstractVector{Float64})
            HL = LowerTriangular(
                Zygote.hessian(lag(obj_factor, lambda), x),
            )
            values .= 0
            add_values!(values, HL)
            return values
        end
    end

    evaluator = JuMPEvaluator(
        nvars, xlb, xub, x0, ineq_nconstr + eq_nconstr, clb, cub,
        njacvals, Hnvalues, obj, eval_g, eval_grad_f, eval_jac_g,
        jac_structure, eval_h, lag_hess_structure,
    )
    jump_model = JuMP.Model(optimizer)
    moi_model = JuMP.backend(jump_model)
    MOI.empty!(moi_model)
    vars = map(1:nvars) do i
        v = MOI.add_variable(moi_model)
        if integers[i]
            if xlb[i] > -1 && xub[i] < 2
                MOI.add_constraint(moi_model, v, MOI.ZeroOne())
            else
                MOI.add_constraint(moi_model, v, MOI.Integer())
            end
        end
        if xub[i] != Inf
            MOI.add_constraint(
                moi_model,
                v,
                MOI.LessThan(xub[i]),
            )
        end
        if xlb[i] != Inf
            MOI.add_constraint(
                moi_model,
                v,
                MOI.GreaterThan(xlb[i]),
            )
        end
        MOI.set(moi_model, MOI.VariablePrimalStart(), v, x0[i])
        return v
    end
    block_data = MOI.NLPBlockData(MOI.NLPBoundsPair.(clb, cub), evaluator, true)
    MOI.set(moi_model, MOI.NLPBlock(), block_data)
    MOI.set(moi_model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return JuMPProblem(evaluator, jump_model, vars)
end

function MOI.initialize(d::JuMPEvaluator, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(d))
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec products for solvers that need them
        end
    end
end

function MOI.features_available(::JuMPEvaluator)
    return [:Grad, :Jac, :Hess]
end

MOI.eval_objective(d::JuMPEvaluator, x) = d.obj(x)

function MOI.eval_constraint(d::JuMPEvaluator, g, x)
    d.eval_g(x, g)
    return
end

function MOI.eval_objective_gradient(d::JuMPEvaluator, grad_f, x)
    d.eval_grad_f(x, grad_f)
    return
end

function MOI.jacobian_structure(d::JuMPEvaluator)
    return d.jac_structure()
end
# lower triangle only
function MOI.hessian_lagrangian_structure(d::JuMPEvaluator)
    return d.lag_hess_structure()
end

function MOI.eval_constraint_jacobian(d::JuMPEvaluator, J, x)
    d.eval_jac_g(x, J)
    return
end

function MOI.eval_hessian_lagrangian(d::JuMPEvaluator, H, x, σ, μ)
    d.eval_h(x, σ, μ, H)
   return
end

nvalues(::Nothing) = 0
nvalues(J::Matrix) = length(J)
nvalues(H::LowerTriangular{<:Real, <:Matrix}) = (size(H, 1) + 1) * size(H, 1) ÷ 2
nvalues(H::SparseMatrixCSC) = length(H.nzval)

# Implement these for sparse matrices
function fill_indices!(rows, cols, J0::Matrix; offset = 0, row_offset = 0)
    nconstr, nvars = size(J0)
    for j in 1:nvars
        cols[offset + 1 : offset + nconstr] .= j
        rows[offset + 1 : offset + nconstr] .= row_offset+1:row_offset+nconstr
        offset += nconstr
    end
    return rows, cols
end
function fill_indices!(rows, cols, HL::LowerTriangular{<:Real, <:Matrix}; offset = 0, row_offset = 0)
    nvars = size(HL, 1)
    for j in 1:nvars
        cols[offset + 1 : offset + nvars - j + 1] .= j
        rows[offset + 1 : offset + nvars - j + 1] .= row_offset + j : row_offset + nvars
        offset += nvars - j + 1
    end
    return rows, cols
end
function fill_indices!(rows, cols, HL::SparseMatrixCSC; offset = 0, row_offset = 0)
    for col in 1:length(HL.colptr)-1
        indices = HL.colptr[col]:HL.colptr[col+1]-1
        nvars = length(indices)
        cols[offset + 1 : offset + nvars] .= col
        rows[offset + 1 : offset + nvars] = row_offset .+ HL.rowval[indices]
        offset += nvars
    end
    return rows, cols
end

function add_values!(values, J::Matrix; offset = 0)
    nvars = length(J)
    values[offset+1:offset+nvars] .+= vec(J)
    return values
end
function add_values!(values, HL::LowerTriangular{<:Real, <:Matrix}; factor = 1, offset = 0)
    nvars = size(HL, 1)
    for j in 1:nvars
        values[offset + 1 : offset + nvars - j + 1] .+= HL[j:nvars, j] .* factor
        offset += nvars - j + 1
    end
    return values
end
function add_values!(values, HL::SparseMatrixCSC; factor = 1, offset = 0)
    nvars = length(HL.nzval)
    values[offset+1:offset+nvars] .= HL.nzval .* factor
    return values
end

_dot(f, x, y) = dot(f(x), y)
_dot(::Nothing, ::Any, ::Any) = 0.0
