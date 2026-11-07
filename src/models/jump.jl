function get_variable_info(model::JuMP.Model)
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
            integer[k] = map(vs) do v
                is_binary(v) || is_integer(v)
            end
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
    else
        error(
            "Unsupported objective function of type $(typeof(obj)). Only affine and " *
            "quadratic objectives can be converted to a DictModel.",
        )
    end
    if sense == MathOptInterface.MAX_SENSE
        return -c, -Q
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

function DictModel(model::JuMP.Model)
    lb, ub, inits, integer = get_variable_info(model)
    nvars = length(flatten(lb)[1])
    Aineq, bineq, Aeq, beq = get_constraint_info(model, nvars)
    c, Q = get_objective_info(model, nvars)
    ks = collect(keys(inits))
    dict_model = DictModel()
    for k in ks
        addvar!(dict_model, k, lb[k], ub[k], init = inits[k], integer = integer[k])
    end
    set_objective!(dict_model, x -> begin
        xv = flatten(x, ks)[1]
        return dot(c, xv) + dot(xv, Q * xv)
    end)
    if length(bineq) > 0
        add_ineq_constraint!(dict_model, x -> Aineq * flatten(x, ks)[1] - bineq)
    end
    if length(beq) > 0
        add_eq_constraint!(dict_model, x -> Aeq * flatten(x, ks)[1] - beq)
    end
    return dict_model
end
