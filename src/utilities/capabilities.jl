######################################################
# Constraint capabilities: every optimizer declares
# which model features it supports, and `optimize`
# validates the (sub)problem against the declaration.
######################################################

"""
    ConstraintCapabilities(; eq = missing, ineq = missing, sd = missing,
                             box = missing, integer = missing)

Declares which constraint/variable features an optimizer supports:

- `eq`: equality constraints,
- `ineq`: inequality constraints,
- `sd`: semidefinite constraints,
- `box`: finite variable bounds,
- `integer`: integer/binary variables.

Each field is one of `true` (supported), `false` (not supported) or `missing`
(unknown/not declared). A solver that returns `nothing` from
[`solver_capabilities`](@ref) declares nothing at all.
"""
struct ConstraintCapabilities
    eq::Union{Bool,Missing}
    ineq::Union{Bool,Missing}
    sd::Union{Bool,Missing}
    box::Union{Bool,Missing}
    integer::Union{Bool,Missing}
end
function ConstraintCapabilities(;
    eq = missing,
    ineq = missing,
    sd = missing,
    box = missing,
    integer = missing,
)
    return ConstraintCapabilities(eq, ineq, sd, box, integer)
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
        eq = d[:eq],
        ineq = d[:ineq],
        sd = d[:sd],
        box = d[:box],
        integer = d[:integer],
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
    assumed::Union{ConstraintCapabilities,Nothing} = nothing,
    hint::String = "",
    context::String = "the model",
)
    caps = assumed === nothing ? solver_capabilities(optimizer) : assumed
    features = Dict{Symbol,Bool}(
        :eq => !isempty(model.eq_constraints.fs),
        :ineq => !isempty(model.ineq_constraints.fs),
        :sd => !isempty(model.sd_constraints.fs),
        :box => any(isfinite, model.box_min) || any(isfinite, model.box_max),
        :integer => any(!iszero, model.integer),
    )
    for f in fieldnames(ConstraintCapabilities)
        features[f] || continue
        v = caps === nothing ? missing : getfield(caps, f)
        v === true && continue
        solvername = string(nameof(typeof(optimizer)))
        msg = "The solver $solvername cannot solve $context which has " *
              string(f) * " " * (
            f === :box ? "bounds on variables" :
            f === :integer ? "integer/binary variables" :
            "$f constraints"
        ) * "."
        if v === false
            msg *= " $solvername does not support this feature."
        else
            msg *= " $solvername must declare whether it supports this " *
                   "feature by specializing `solver_capabilities`."
        end
        isempty(hint) || (msg *= " " * hint)
        throw(ArgumentError(msg))
    end
    return nothing
end
