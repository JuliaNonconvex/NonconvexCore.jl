######################################################
# Constraint capabilities: every optimizer declares
# which model features it supports, and `optimize`
# validates the (sub)problem against the declaration.
######################################################

"""
    ConstraintCapabilities(; linear_eq = missing, nonlinear_eq = missing,
                             linear_ineq = missing, nonlinear_ineq = missing,
                             linear_sd = missing, nonlinear_sd = missing,
                             sets_eq = false, sets_ineq = false,
                             sets_sd = false, sets_simplex = false,
                             sets_product = false, variable_sets_sd = false,
                             variable_sets_simplex = false, box = missing,
                             continuous = missing, binary = missing,
                             integer = missing, infinite_bounds = missing)

Declares which constraint/variable features an optimizer supports:

- `linear_eq`: linear equality constraints,
- `nonlinear_eq`: nonlinear equality constraints,
- `linear_ineq`: linear inequality constraints,
- `nonlinear_ineq`: nonlinear inequality constraints,
- `linear_sd`: linear semidefinite constraints,
- `nonlinear_sd`: nonlinear semidefinite constraints,
- `sets_eq`: function-in-set equality constraints (`EqualTo`, `Zeros`,
  defaults to `false` — not supported),
- `sets_ineq`: function-in-set inequality/order constraints (`LessThan`,
  `GreaterThan`, `Interval`, `Nonnegatives`, `Nonpositives`; defaults to
  `false`),
- `sets_sd`: function-in-set semidefinite constraints (`PSDCone`; defaults
  to `false`),
- `sets_simplex`: function-in-set simplex constraints (`Simplex`; defaults
  to `false`),
- `sets_product`: function-in-set product constraints (`SetProduct`;
  defaults to `false`),
- `sets_chance`: probabilistic (chance) constraints (`ChanceSet`, including
  per-output probabilities and deterministic `Zero()` constraints; defaults
  to `false`),
- `variable_sets_sd`: variables defined as `PSDCone` blocks (semidefinite
  matrix variables; defaults to `false`),
- `variable_sets_simplex`: variables defined as `Simplex` blocks (defaults
  to `false`),
- `box`: finite variable bounds,
- `continuous`: continuous variables,
- `binary`: binary (0/1) variables,
- `integer`: integer (non-binary) variables,
- `infinite_bounds`: variables with infinite (unbounded) bounds.

Each field is one of `true` (supported), `false` (not supported) or `missing`
(unknown/not declared). A solver that returns `nothing` from
[`solver_capabilities`](@ref) declares nothing at all.
"""
struct ConstraintCapabilities
    linear_eq::Union{Bool,Missing}
    nonlinear_eq::Union{Bool,Missing}
    linear_ineq::Union{Bool,Missing}
    nonlinear_ineq::Union{Bool,Missing}
    linear_sd::Union{Bool,Missing}
    nonlinear_sd::Union{Bool,Missing}
    sets_eq::Union{Bool,Missing}
    sets_ineq::Union{Bool,Missing}
    sets_sd::Union{Bool,Missing}
    sets_simplex::Union{Bool,Missing}
    sets_product::Union{Bool,Missing}
    sets_chance::Union{Bool,Missing}
    variable_sets_sd::Union{Bool,Missing}
    variable_sets_simplex::Union{Bool,Missing}
    box::Union{Bool,Missing}
    continuous::Union{Bool,Missing}
    binary::Union{Bool,Missing}
    integer::Union{Bool,Missing}
    infinite_bounds::Union{Bool,Missing}
end
function ConstraintCapabilities(;
    linear_eq=missing,
    nonlinear_eq=missing,
    linear_ineq=missing,
    nonlinear_ineq=missing,
    linear_sd=missing,
    nonlinear_sd=missing,
    sets_eq=false,
    sets_ineq=false,
    sets_sd=false,
    sets_simplex=false,
    sets_product=false,
    sets_chance=false,
    variable_sets_sd=false,
    variable_sets_simplex=false,
    box=missing,
    continuous=missing,
    binary=missing,
    integer=missing,
    infinite_bounds=missing,
)
    return ConstraintCapabilities(
        linear_eq,
        nonlinear_eq,
        linear_ineq,
        nonlinear_ineq,
        linear_sd,
        nonlinear_sd,
        sets_eq,
        sets_ineq,
        sets_sd,
        sets_simplex,
        sets_product,
        sets_chance,
        variable_sets_sd,
        variable_sets_simplex,
        box,
        continuous,
        binary,
        integer,
        infinite_bounds,
    )
end

# Backward-compatible positional constructor (the pre-2.3 field order): the
# set capabilities default to `false` (not supported).
function ConstraintCapabilities(
    linear_eq::Union{Bool,Missing},
    nonlinear_eq::Union{Bool,Missing},
    linear_ineq::Union{Bool,Missing},
    nonlinear_ineq::Union{Bool,Missing},
    linear_sd::Union{Bool,Missing},
    nonlinear_sd::Union{Bool,Missing},
    box::Union{Bool,Missing},
    continuous::Union{Bool,Missing},
    binary::Union{Bool,Missing},
    integer::Union{Bool,Missing},
    infinite_bounds::Union{Bool,Missing},
)
    return ConstraintCapabilities(;
        linear_eq=linear_eq,
        nonlinear_eq=nonlinear_eq,
        linear_ineq=linear_ineq,
        nonlinear_ineq=nonlinear_ineq,
        linear_sd=linear_sd,
        nonlinear_sd=nonlinear_sd,
        box=box,
        continuous=continuous,
        binary=binary,
        integer=integer,
        infinite_bounds=infinite_bounds,
    )
end

"""
    solver_capabilities(::AbstractOptimizer)

Returns the [`ConstraintCapabilities`](@ref) of the optimizer, or `nothing`
when the solver does not declare capabilities. Every optimizer that can
receive constrained models must specialize this method: `optimize` validates
the model against the declaration via [`checksubproblem!`](@ref) and errors
when capabilities are undeclared or unsupported.
"""
solver_capabilities(::AbstractOptimizer) = nothing

"""
    merge_capabilities(base::Union{ConstraintCapabilities,Nothing}; kwargs...)

Returns a new `ConstraintCapabilities` where fields not overridden by `kwargs`
are taken from `base` (`nothing` fields become `missing`). Useful for
meta-algorithms that forward part of their sub-solver's capabilities.
"""
function merge_capabilities(base::Union{ConstraintCapabilities,Nothing}; kwargs...)
    d = Dict{Symbol,Union{Bool,Missing}}(
        f => missing for f in fieldnames(ConstraintCapabilities)
    )
    if base !== nothing
        for f in fieldnames(ConstraintCapabilities)
            v = getfield(base, f)
            v === missing || (d[f] = v)
        end
    end
    merge!(d, Dict{Symbol,Union{Bool,Missing}}(kwargs))
    return ConstraintCapabilities(;
        linear_eq=d[:linear_eq],
        nonlinear_eq=d[:nonlinear_eq],
        linear_ineq=d[:linear_ineq],
        nonlinear_ineq=d[:nonlinear_ineq],
        linear_sd=d[:linear_sd],
        nonlinear_sd=d[:nonlinear_sd],
        sets_eq=d[:sets_eq],
        sets_ineq=d[:sets_ineq],
        sets_sd=d[:sets_sd],
        sets_simplex=d[:sets_simplex],
        sets_product=d[:sets_product],
        sets_chance=d[:sets_chance],
        variable_sets_sd=d[:variable_sets_sd],
        variable_sets_simplex=d[:variable_sets_simplex],
        box=d[:box],
        continuous=d[:continuous],
        binary=d[:binary],
        integer=d[:integer],
        infinite_bounds=d[:infinite_bounds],
    )
end

function _capability_string(feature::Symbol, v::Union{Bool,Missing})
    s = string(feature)
    if v === missing
        return "$s constraints (not declared)"
    elseif v
        return "$s constraints (supported)"
    else
        return "$s constraints (unsupported)"
    end
end

"""
    checksubproblem!(model::VecModel, optimizer; assumed = nothing,
                     hint = "", context = "the model")

Validates the features present in `model` against
[`solver_capabilities`](@ref) of `optimizer`. Semantics per present feature:

- declared `true` → pass;
- declared `false` → `ArgumentError` naming the solver, the feature and a
  hint on how to work around it;
- undeclared (`missing` or no declaration at all) → `ArgumentError` requiring
  the solver to declare the capability via `solver_capabilities`
  (strict-by-default: no opt-in).

Keyword arguments:

- `assumed`: optional `ConstraintCapabilities` to use instead of querying the
  solver (for meta-algorithms that override their sub-solver's declaration),
- `hint`: extra text appended to error messages,
- `context`: description of the model used in error messages.
"""
function checksubproblem!(
    model::VecModel,
    optimizer::AbstractOptimizer;
    assumed::Union{ConstraintCapabilities,Nothing}=nothing,
    hint::String="",
    context::String="the model",
)
    caps = assumed === nothing ? solver_capabilities(optimizer) : assumed
    x = getinit(model)
    eq_lin = _constraint_linearity(model.eq_constraints, x)
    ineq_lin = _constraint_linearity(model.ineq_constraints, x)
    sd_lin = _constraint_linearity(model.sd_constraints, x)
    n = length(model.integer)
    discrete = BitVector(undef, n)
    binary_flag = BitVector(undef, n)
    for i in 1:n
        discrete[i] = model.integer[i] != 0
        binary_flag[i] = discrete[i] && model.box_min[i] >= 0 && model.box_max[i] <= 1
    end
    # Function-in-set features, split per set kind. The cone parts of set
    # variables are enforced as synthetic set constraints on the block (a PSD
    # constraint for PSDCone blocks, a sum-to-one equality for simplex
    # blocks), so the matching `sets_*` capabilities fire through the
    # synthetic constraints themselves; the variable blocks additionally
    # require the corresponding `variable_sets_*` capability.
    set_features = Dict{Symbol,Bool}(
        :sets_eq => false,
        :sets_ineq => false,
        :sets_sd => false,
        :sets_simplex => false,
        :sets_product => false,
        :sets_chance => false,
    )
    var_features = Dict{Symbol,Bool}(
        :variable_sets_sd => false, :variable_sets_simplex => false
    )
    for c in model.set_constraints.fs
        for k in set_kinds(c.set)
            set_features[Symbol(:sets_, k)] = true
        end
    end
    for v in model.set_variables
        for k in variable_set_kinds(v.set)
            var_features[Symbol(:variable_sets_, k)] = true
        end
    end
    features = Dict{Symbol,Bool}(
        :linear_eq => any(eq_lin),
        :nonlinear_eq => any(.!eq_lin),
        :linear_ineq => any(ineq_lin),
        :nonlinear_ineq => any(.!ineq_lin),
        :linear_sd => any(sd_lin),
        :nonlinear_sd => any(.!sd_lin),
        set_features...,
        var_features...,
        :box => any(isfinite, model.box_min) || any(isfinite, model.box_max),
        :continuous => any(.!discrete),
        :binary => any(binary_flag),
        :integer => any(discrete .&& .!binary_flag),
        :infinite_bounds => any(!isfinite, model.box_min) || any(!isfinite, model.box_max),
    )
    featurenames = Dict{Symbol,String}(
        :linear_eq => "linear equality constraints",
        :nonlinear_eq => "nonlinear equality constraints",
        :linear_ineq => "linear inequality constraints",
        :nonlinear_ineq => "nonlinear inequality constraints",
        :linear_sd => "linear semidefinite constraints",
        :nonlinear_sd => "nonlinear semidefinite constraints",
        :sets_eq => "function-in-set equality constraints (EqualTo)",
        :sets_ineq => "function-in-set inequality constraints (LessThan/GreaterThan/Interval/Nonnegatives/Nonpositives)",
        :sets_sd => "function-in-set semidefinite constraints (PSDCone)",
        :sets_simplex => "function-in-set simplex constraints (Simplex)",
        :sets_product => "function-in-set product constraints (SetProduct)",
        :sets_chance => "probabilistic (chance) constraints (ChanceSet)",
        :variable_sets_sd => "semidefinite (PSDCone) variable blocks",
        :variable_sets_simplex => "simplex variable blocks",
        :box => "bounds on variables",
        :continuous => "continuous variables",
        :binary => "binary variables",
        :integer => "integer variables",
        :infinite_bounds => "infinite (unbounded) variable bounds",
    )
    for f in fieldnames(ConstraintCapabilities)
        features[f] || continue
        v = caps === nothing ? missing : getfield(caps, f)
        v === true && continue
        solvername = string(nameof(typeof(optimizer)))
        msg =
            "The solver $solvername cannot solve $context which has " *
            featurenames[f] *
            "."
        if v === false
            msg *= " $solvername does not support this feature."
        else
            msg *=
                " $solvername must declare whether it supports this " *
                "feature by specializing `solver_capabilities`."
        end
        isempty(hint) || (msg *= " " * hint)
        throw(ArgumentError(msg))
    end
    return nothing
end

# Classify the constraints of a collection as linear (true) or nonlinear
# (false) per output row. JuMP-converted constraints use the linearity of
# their expression graphs (exact); any other constraint is classified by its
# structural Hessian: a function is structurally linear exactly when its
# Hessian pattern is empty. Functions that cannot be traced are
# conservatively classified as nonlinear.
function _constraint_linearity(cs::VectorOfFunctions, x)
    out = BitVector()
    for c in cs.fs
        raw = c.f isa FlattenedFunction ? getrawfunction(c.f) : c.f
        raw = raw isa FunctionWrapper ? getfunction(raw) : raw
        # First-class linear constraints are classified exactly.
        inner = raw isa FunctionInSet ? getfunction(raw) : raw
        if inner isa LinearFunction
            append!(out, fill(true, getdim(c)))
        elseif raw isa JuMPModelIneq
            lin, _ = jump_constraint_linearity(raw.m)
            append!(out, lin)
        elseif raw isa JuMPModelEq
            _, lin = jump_constraint_linearity(raw.m)
            append!(out, lin)
        else
            m = getdim(c)
            lin = try
                H = hessian_sparsity(c.f, x, TracerSparsityDetector())
                all(iszero, H)
            catch
                false
            end
            append!(out, fill(lin, m))
        end
    end
    return out
end
