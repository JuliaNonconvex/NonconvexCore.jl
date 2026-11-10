module NonconvexCoreJuMPExt

using NonconvexCore: NonconvexCore
using JuMP: JuMP
using MathOptInterface: MathOptInterface
const MOI = MathOptInterface
using SparseArrays: sparse, sparsevec, spzeros
using LinearAlgebra: dot, LowerTriangular, mul!
using NonconvexCore:
    DictModel,
    addvar!,
    set_objective!,
    add_ineq_constraint!,
    add_eq_constraint!,
    flatten,
    OrderedDict,
    JuMPEvaluator,
    JuMPModelObjective,
    JuMPModelIneq,
    JuMPModelEq,
    get_variable_info,
    _ordered_jump_variables,
    _nonlinear_constraints,
    VecModel,
    getinit,
    CountingFunction,
    getobjective,
    getmin,
    getmax,
    DEFAULT_AD_BACKEND,
    AbstractADType,
    sparse_jacobian,
    sparse_hessian,
    nvalues,
    fill_indices!,
    add_values!,
    VectorOfFunctions,
    get_jump_problem,
    _dot,
    value_jacobian,
    JuMPProblem,
    JuMPProblemEvaluator
using DifferentiationInterface: DifferentiationInterface
const DI = DifferentiationInterface
using JuMP:
    VariableRef,
    is_binary,
    is_integer,
    has_lower_bound,
    has_upper_bound,
    lower_bound,
    upper_bound,
    start_value,
    ConstraintRef,
    constraint_object,
    AffExpr,
    QuadExpr,
    objective_function,
    objective_sense

function NonconvexCore.get_variable_info(model::JuMP.Model)
    varinds = OrderedDict()
    integer = OrderedDict()
    lb = OrderedDict()
    ub = OrderedDict()
    inits = OrderedDict()
    for (k, vs) in model.obj_dict
        if vs isa VariableRef
            v = vs
            ind = v.index.value
            varinds[k] = ind
            integer[k] = is_binary(v) || is_integer(v)
            lb[k] = if has_lower_bound(v)
                lower_bound(v)
            elseif is_binary(v)
                0.0
            else
                -Inf
            end
            ub[k] = if has_upper_bound(v)
                upper_bound(v)
            elseif is_binary(v)
                1.0
            else
                Inf
            end
            inits[k] = if start_value(v) !== nothing
                start_value(v)
            elseif ub[k] == Inf && lb[k] == -Inf
                0.0
            elseif ub[k] == Inf
                lb[k] + 1.0
            elseif lb[k] == -Inf
                ub[k] - 1.0
            else
                (ub[k] + lb[k]) / 2
            end
        elseif vs isa AbstractArray{<:VariableRef}
            inds = map(v -> v.index.value, vs)
            if inds isa JuMP.Containers.DenseAxisArray
                varinds[k] = OrderedDict(inds.data .=> keys(inds))
            else
                varinds[k] = OrderedDict(vec(inds) .=> 1:length(inds))
            end
            # Store the integer flags as a flat BitVector: the bounds keep the
            # structured form, but flattening heterogeneous array-typed dict
            # values is not supported by the flattening machinery.
            ints = map(vs) do v
                return is_binary(v) || is_integer(v)
            end
            integer[k] = BitVector(
                vec(ints isa JuMP.Containers.DenseAxisArray ? ints.data : collect(ints))
            )
            lb[k] = map(vs) do v
                if has_lower_bound(v)
                    lower_bound(v)
                elseif is_binary(v)
                    0.0
                else
                    -Inf
                end
            end
            ub[k] = map(vs) do v
                if has_upper_bound(v)
                    upper_bound(v)
                elseif is_binary(v)
                    1.0
                else
                    Inf
                end
            end
            inits[k] = map(vs) do v
                ind = varinds[k][v.index.value]
                if start_value(v) !== nothing
                    start_value(v)
                elseif ub[k][ind] == Inf && lb[k][ind] == -Inf
                    0.0
                elseif ub[k][ind] == Inf
                    lb[k][ind] + 1.0
                elseif lb[k][ind] == -Inf
                    ub[k][ind] - 1.0
                else
                    (ub[k][ind] + lb[k][ind]) / 2
                end
            end
        else
            continue
        end
    end
    sort!(integer)
    sort!(lb)
    sort!(ub)
    sort!(inits)
    _lb = OrderedDict(k => v for (k, v) in lb)
    _ub = OrderedDict(k => v for (k, v) in ub)
    _inits = OrderedDict(k => v for (k, v) in inits)
    _integer = OrderedDict(k => v for (k, v) in integer)
    return _lb, _ub, _inits, _integer
end

function get_constraint_info(model::JuMP.Model, nvars)
    Iineq = Int[]
    Jineq = Int[]
    Vineq = Float64[]
    bineq = Float64[]

    Ieq = Int[]
    Jeq = Int[]
    Veq = Float64[]
    beq = Float64[]

    ineqcounter = 0
    eqcounter = 0
    inv_geq_constraints = OrderedDict()
    inv_leq_constraints = OrderedDict()
    inv_eq_constraints = OrderedDict()
    for (_, c) in model.obj_dict
        if c isa ConstraintRef
            cs = [c]
        elseif c isa AbstractArray{<:ConstraintRef}
            cs = c
        else
            cs = nothing
            continue
        end
        foreach(cs) do c
            set = constraint_object(c).set
            if set isa MOI.GreaterThan
                inv_geq_constraints[JuMP.index(c).value] = c
            elseif set isa MOI.LessThan
                inv_leq_constraints[JuMP.index(c).value] = c
            elseif set isa MOI.EqualTo
                inv_eq_constraints[JuMP.index(c).value] = c
            else
                throw("Set type not supported.")
            end
        end
    end
    sort!(inv_geq_constraints)
    sort!(inv_leq_constraints)
    sort!(inv_eq_constraints)
    geq_constraints = values(inv_geq_constraints)
    leq_constraints = values(inv_leq_constraints)
    eq_constraints = values(inv_eq_constraints)

    for c in geq_constraints
        func = constraint_object(c).func
        @assert func isa AffExpr
        set = constraint_object(c).set
        ineqcounter += 1
        push!(bineq, -set.lower)
        for (var, val) in func.terms
            push!(Iineq, ineqcounter)
            push!(Jineq, var.index.value)
            push!(Vineq, -val)
        end
    end
    for c in leq_constraints
        func = constraint_object(c).func
        @assert func isa AffExpr
        set = constraint_object(c).set
        ineqcounter += 1
        push!(bineq, set.upper)
        for (var, val) in func.terms
            push!(Iineq, ineqcounter)
            push!(Jineq, var.index.value)
            push!(Vineq, val)
        end
    end
    for c in eq_constraints
        func = constraint_object(c).func
        @assert func isa AffExpr
        set = constraint_object(c).set
        eqcounter += 1
        push!(beq, set.value)
        for (var, val) in func.terms
            push!(Ieq, eqcounter)
            push!(Jeq, var.index.value)
            push!(Veq, val)
        end
    end
    Aineq = sparse(Iineq, Jineq, Vineq, ineqcounter, nvars)
    Aeq = sparse(Ieq, Jeq, Veq, eqcounter, nvars)

    return Aineq, bineq, Aeq, beq
end

function get_objective_info(model::JuMP.Model, nvars)
    obj = objective_function(model)
    sense = objective_sense(model)
    if obj isa AffExpr
        c, Q = _affine_objective(obj, nvars)
    elseif obj isa QuadExpr
        c, Q = _quadratic_objective(obj, nvars)
    elseif obj isa MOI.ScalarNonlinearFunction || obj isa JuMP.GenericNonlinearExpr
        c, Q = nothing, nothing
    else
        error(
            "Unsupported objective function of type $(typeof(obj)). Only affine, " *
            "quadratic and nonlinear (ScalarNonlinearFunction) objectives can " *
            "be converted to a DictModel.",
        )
    end
    if sense == MathOptInterface.MAX_SENSE
        return c === nothing ? (nothing, nothing) : (-c, -Q)
    else
        return c, Q
    end
end

function _affine_objective(obj::AffExpr, nvars)
    I = Int[]
    V = Float64[]
    for (var, val) in obj.terms
        push!(I, var.index.value)
        push!(V, val)
    end
    return sparsevec(I, V, nvars), spzeros(nvars, nvars)
end

function _quadratic_objective(obj::QuadExpr, nvars)
    c, _ = _affine_objective(obj.aff, nvars)
    II = Int[]
    JJ = Int[]
    V = Float64[]
    for (pair, coef) in obj.terms
        i, j = pair.a.index.value, pair.b.index.value
        if i == j
            push!(II, i)
            push!(JJ, j)
            push!(V, coef)
        else
            # Split the cross term so Q is symmetric and x'Qx reproduces
            # sum(coef * x[i] * x[j]) over the unordered pairs.
            push!(II, i)
            push!(JJ, j)
            push!(V, coef / 2)
            push!(II, j)
            push!(JJ, i)
            push!(V, coef / 2)
        end
    end
    return c, sparse(II, JJ, V, nvars, nvars)
end

function NonconvexCore.DictModel(model::JuMP.Model)
    lb, ub, inits, integer = get_variable_info(model)
    nvars = length(flatten(lb)[1])
    ks = collect(keys(inits))
    dict_model = DictModel()
    for k in ks
        addvar!(dict_model, k, lb[k], ub[k]; init=inits[k], integer=integer[k])
    end
    if _has_nonlinear(model)
        # Uniform expression path: every objective and constraint is evaluated
        # through MathOptInterface's Nonlinear evaluator, whose direct
        # derivatives apply to the whole model.
        return _dict_model_from_jump_expressions(model, dict_model, ks, nvars)
    end
    Aineq, bineq, Aeq, beq = get_constraint_info(model, nvars)
    c, Q = get_objective_info(model, nvars)
    set_objective!(dict_model, x -> begin
        xv = flatten(x, ks)[1]
        # `c` is dense in memory: a sparse vector dotted with a tracked
        # vector is method-ambiguous for some reverse-mode backends.
        return dot(Vector(c), xv) + dot(xv, Q * xv)
    end)
    if length(bineq) > 0
        add_ineq_constraint!(dict_model, x -> Aineq * flatten(x, ks)[1] - bineq)
    end
    if length(beq) > 0
        add_eq_constraint!(dict_model, x -> Aeq * flatten(x, ks)[1] - beq)
    end
    return dict_model
end

# The JuMP variables ordered like the flattened model vector. The indices are
# flattened with the same machinery as the model values, so the order matches.
function NonconvexCore._ordered_jump_variables(model::JuMP.Model, ks)
    inds = OrderedDict{Any,Any}()
    for (k, vs) in model.obj_dict
        if vs isa VariableRef
            inds[k] = vs.index.value
        elseif vs isa AbstractArray{<:VariableRef}
            inds[k] = map(v -> v.index.value, vs)
        end
    end
    return MOI.VariableIndex.(flatten(OrderedDict(k => inds[k] for k in ks))[1])
end

# Nonlinear (ScalarNonlinearFunction) constraints with their sets.
function NonconvexCore._nonlinear_constraints(model::JuMP.Model)
    out = Tuple{MOI.ScalarNonlinearFunction,Any}[]
    for c in JuMP.all_constraints(model; include_variable_in_set_constraints=true)
        func = constraint_object(c).func
        func isa JuMP.GenericNonlinearExpr && (func = JuMP.moi_function(func))
        func isa QuadExpr && (func = JuMP.moi_function(func))
        func isa AffExpr && (func = JuMP.moi_function(func))
        func isa MOI.AbstractScalarFunction && push!(out, (func, constraint_object(c).set))
    end
    return out
end

function _has_nonlinear(model::JuMP.Model)
    objective_function(model) isa JuMP.GenericNonlinearExpr && return true
    return any(JuMP.all_constraints(model; include_variable_in_set_constraints=true)) do c
        return constraint_object(c).func isa JuMP.GenericNonlinearExpr
    end
end

function _dict_model_from_jump_expressions(model, dict_model, ks, nvars)
    vars = _ordered_jump_variables(model, ks)
    # One Nonlinear model for the objective and every constraint, so the
    # evaluator computes shared subexpressions once per pass.
    nl = MOI.Nonlinear.Model()
    obj_expr = objective_function(model)
    if obj_expr isa JuMP.GenericNonlinearExpr ||
        obj_expr isa QuadExpr ||
        obj_expr isa AffExpr
        obj_expr = JuMP.moi_function(obj_expr)
    end
    MOI.Nonlinear.set_objective(nl, obj_expr)
    obj_sign = objective_sense(model) == MathOptInterface.MAX_SENSE ? -1.0 : 1.0
    ineq_cis = MOI.Nonlinear.ConstraintIndex[]
    ineq_sign = Float64[]
    ineq_shift = Float64[]
    ineq_linear = Bool[]
    eq_cis = MOI.Nonlinear.ConstraintIndex[]
    eq_shift = Float64[]
    eq_linear = Bool[]
    # Nonconvex inequality constraints are feasible when their value is <= 0,
    # so each row stores `sign * f(x) + shift`. JuMP may have folded the
    # bound into the expression (its nonlinear interface normalizes to zero
    # sets); carrying the raw bound works under both conventions.
    for (expr, set) in _nonlinear_constraints(model)
        linear = expr isa MOI.ScalarAffineFunction
        if set isa MOI.GreaterThan
            push!(ineq_cis, MOI.Nonlinear.add_constraint(nl, expr, set))
            push!(ineq_sign, -1.0)
            push!(ineq_shift, set.lower)
            push!(ineq_linear, linear)
        elseif set isa MOI.LessThan
            push!(ineq_cis, MOI.Nonlinear.add_constraint(nl, expr, set))
            push!(ineq_sign, 1.0)
            push!(ineq_shift, -set.upper)
            push!(ineq_linear, linear)
        elseif set isa MOI.Interval
            push!(
                ineq_cis, MOI.Nonlinear.add_constraint(nl, expr, MOI.GreaterThan(set.lower))
            )
            push!(ineq_sign, -1.0)
            push!(ineq_shift, set.lower)
            push!(ineq_linear, linear)
            push!(ineq_cis, MOI.Nonlinear.add_constraint(nl, expr, MOI.LessThan(set.upper)))
            push!(ineq_sign, 1.0)
            push!(ineq_shift, -set.upper)
            push!(ineq_linear, linear)
        elseif set isa MOI.EqualTo
            push!(eq_cis, MOI.Nonlinear.add_constraint(nl, expr, set))
            push!(eq_shift, -set.value)
            push!(eq_linear, linear)
        end
    end
    je = JuMPEvaluator(
        nl,
        vars,
        obj_sign,
        ineq_cis,
        ineq_sign,
        ineq_shift,
        eq_cis,
        eq_shift,
        ineq_linear,
        eq_linear,
    )
    set_objective!(dict_model, JuMPModelObjective(je, ks))
    je.nineq > 0 && add_ineq_constraint!(dict_model, JuMPModelIneq(je, ks))
    je.neq > 0 && add_eq_constraint!(dict_model, JuMPModelEq(je, ks))
    return dict_model
end

function NonconvexCore.get_jump_problem(
    model::VecModel,
    x0=getinit(model);
    optimizer,
    first_order,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
    kwargs...,
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
        obj,
        ineq,
        eq,
        x0,
        integers,
        getmin(model),
        getmax(model),
        first_order,
        optimizer,
        backend,
    ),
    obj.counter
end
function NonconvexCore.get_jump_problem(
    obj,
    ineq_constr,
    eq_constr,
    x0,
    integers,
    xlb,
    xub,
    first_order,
    optimizer,
    backend::AbstractADType=DEFAULT_AD_BACKEND[],
)
    nvars = length(x0)
    if ineq_constr !== nothing
        ineqJ0 = sparse_jacobian(ineq_constr, x0, backend)
        ineq_nconstr = size(ineqJ0, 1)
        Joffset = nvalues(ineqJ0)
    else
        ineqJ0 = nothing
        ineq_nconstr = 0
        Joffset = 0
    end
    if eq_constr !== nothing
        eqJ0 = sparse_jacobian(eq_constr, x0, backend)
        eq_nconstr = size(eqJ0, 1)
    else
        eqJ0 = nothing
        eq_nconstr = 0
    end
    njacvals = nvalues(ineqJ0) + nvalues(eqJ0)
    @assert nvars > 0
    # The Jacobian structure is detected once at `x0` with
    # SparseConnectivityTracer and stays fixed for the whole solve.
    jac_rows = zeros(Int, njacvals)
    jac_cols = zeros(Int, njacvals)
    ineqJ0 === nothing || fill_indices!(jac_rows, jac_cols, ineqJ0)
    eqJ0 === nothing ||
        fill_indices!(jac_rows, jac_cols, eqJ0; offset=Joffset, row_offset=ineq_nconstr)
    jac_structure_data = tuple.(jac_rows, jac_cols)
    lag(factor, y) =
        x -> begin
            factor * obj(x) +
            _dot(ineq_constr, x, @view(y[1:ineq_nconstr])) +
            _dot(eq_constr, x, @view(y[(ineq_nconstr + 1):end]))
        end
    clb = [fill(-Inf, ineq_nconstr); zeros(eq_nconstr)]
    cub = zeros(ineq_nconstr + eq_nconstr)

    function eval_g(x::AbstractVector{Float64}, g::AbstractVector{Float64})
        if ineq_constr !== nothing
            g[1:ineq_nconstr] .= ineq_constr(x)
        end
        if eq_constr !== nothing
            g[(ineq_nconstr + 1):end] .= eq_constr(x)
        end
        return g
    end
    function eval_grad_f(x::AbstractVector{Float64}, grad_f::AbstractVector{Float64})
        return grad_f .= DI.gradient(obj, backend, x)
    end
    jac_structure() = jac_structure_data
    function eval_jac_g(x::AbstractVector{Float64}, values::AbstractVector{Float64})
        if ineq_constr !== nothing
            J = Matrix(value_jacobian(ineq_constr, x, backend)[2])
            for k in 1:Joffset
                values[k] = J[jac_rows[k], jac_cols[k]]
            end
        end
        if eq_constr !== nothing
            J = Matrix(value_jacobian(eq_constr, x, backend)[2])
            for k in (Joffset + 1):njacvals
                values[k] = J[jac_rows[k] - ineq_nconstr, jac_cols[k]]
            end
        end
        return values
    end

    if first_order
        eval_h = (x...) -> 0.0
        lag_hess_structure = () -> Tuple{Int,Int}[]
        Hnvalues = 0
    else
        # The Hessian structure is detected once at `x0` with all constraint
        # multipliers active so that it contains every possible entry.
        HL0 = LowerTriangular(
            sparse_hessian(lag(1.0, ones(ineq_nconstr + eq_nconstr)), x0, backend)
        )
        Hnvalues = nvalues(HL0)
        h_rows = zeros(Int, Hnvalues)
        h_cols = zeros(Int, Hnvalues)
        fill_indices!(h_rows, h_cols, HL0)
        lag_hess_structure_data = tuple.(h_rows, h_cols)
        lag_hess_structure = () -> lag_hess_structure_data
        eval_h = function (
            x::AbstractVector{Float64},
            obj_factor::Float64,
            lambda::AbstractVector{Float64},
            values::AbstractVector{Float64},
        )
            H = DI.hessian(lag(obj_factor, lambda), backend, x)
            for k in 1:Hnvalues
                values[k] = H[h_rows[k], h_cols[k]]
            end
            return values
        end
    end

    evaluator = JuMPProblemEvaluator(
        nvars,
        xlb,
        xub,
        x0,
        ineq_nconstr + eq_nconstr,
        clb,
        cub,
        njacvals,
        Hnvalues,
        obj,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        jac_structure,
        eval_h,
        lag_hess_structure,
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
            MOI.add_constraint(moi_model, v, MOI.LessThan(xub[i]))
        end
        if xlb[i] != -Inf
            MOI.add_constraint(moi_model, v, MOI.GreaterThan(xlb[i]))
        end
        MOI.set(moi_model, MOI.VariablePrimalStart(), v, x0[i])
        return v
    end
    block_data = MOI.NLPBlockData(MOI.NLPBoundsPair.(clb, cub), evaluator, true)
    MOI.set(moi_model, MOI.NLPBlock(), block_data)
    MOI.set(moi_model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return JuMPProblem(evaluator, jump_model, vars)
end

function MOI.initialize(d::JuMPProblemEvaluator, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(d))
            error("Unsupported feature $feat")
        end
    end
end

function MOI.features_available(::JuMPProblemEvaluator)
    return [:Grad, :Jac, :Hess, :JacVec, :HessVec]
end

MOI.eval_objective(d::JuMPProblemEvaluator, x) = d.obj(x)

function MOI.eval_constraint(d::JuMPProblemEvaluator, g, x)
    d.eval_g(x, g)
    return nothing
end

function MOI.eval_objective_gradient(d::JuMPProblemEvaluator, grad_f, x)
    d.eval_grad_f(x, grad_f)
    return nothing
end

function MOI.jacobian_structure(d::JuMPProblemEvaluator)
    return d.jac_structure()
end
# lower triangle only
function MOI.hessian_lagrangian_structure(d::JuMPProblemEvaluator)
    return d.lag_hess_structure()
end

function MOI.eval_constraint_jacobian(d::JuMPProblemEvaluator, J, x)
    d.eval_jac_g(x, J)
    return nothing
end

function MOI.eval_hessian_lagrangian(d::JuMPProblemEvaluator, H, x, σ, μ)
    d.eval_h(x, σ, μ, H)
    return nothing
end

# Jacobian-vector and Hessian-vector products, computed from the full sparse
# Jacobian and Lagrangian Hessian.

function _dense_jacobian(d::JuMPProblemEvaluator, x)
    structure = d.jac_structure()
    vals = zeros(d.njacvalues)
    d.eval_jac_g(x, vals)
    J = zeros(d.nconstr, d.nvars)
    for (k, (row, col)) in enumerate(structure)
        J[row, col] = vals[k]
    end
    return J
end

function MOI.eval_constraint_jacobian_product(d::JuMPProblemEvaluator, y, x, w)
    mul!(y, _dense_jacobian(d, x), w)
    return y
end

function MOI.eval_constraint_jacobian_transpose_product(d::JuMPProblemEvaluator, y, x, w)
    mul!(y, transpose(_dense_jacobian(d, x)), w)
    return y
end

function _dense_hessian(d::JuMPProblemEvaluator, x, σ, μ)
    structure = d.lag_hess_structure()
    vals = zeros(d.nhessvalues)
    d.eval_h(x, σ, μ, vals)
    H = zeros(d.nvars, d.nvars)
    for (k, (i, j)) in enumerate(structure)
        H[i, j] = vals[k]
        H[j, i] = vals[k]
    end
    return H
end

function MOI.eval_hessian_lagrangian_product(d::JuMPProblemEvaluator, h, x, v, σ, μ)
    mul!(h, _dense_hessian(d, x, σ, μ), v)
    return h
end

end # module NonconvexCoreJuMPExt
