"""
    TraceFunction(f; on_call = ..., on_grad = ...)

Wraps `f` and records a trace of where it is evaluated (on the function call
and/or on gradient evaluations), for debugging which part of a model calls a
function during optimization.
"""
struct TraceFunction{F,V} <: Function
    f::F
    trace::V
    on_call::Bool
    on_grad::Bool
end
function TraceFunction(
    f; on_call::Union{Bool,Nothing}=nothing, on_grad::Union{Bool,Nothing}=nothing
)
    if on_call === on_grad === nothing
        _on_call = true
        _on_grad = true
    elseif on_call === nothing
        _on_call = !something(on_grad, true)
        _on_grad = on_grad
    elseif on_grad === nothing
        _on_call = on_call
        _on_grad = !something(on_call, true)
    else
        _on_call = on_call
        _on_grad = on_grad
    end
    return TraceFunction(f, Any[], _on_call::Bool, _on_grad::Bool)
end
function (tf::TraceFunction)(x)
    v = tf.f(x)
    if tf.on_call
        push!(tf.trace, (input=copy(x), output=copy(v)))
    end
    return v
end
function ChainRulesCore.rrule(rc::RuleConfig, tf::TraceFunction, x)
    v, pb = ChainRulesCore.rrule_via_ad(rc, tf.f, x)
    return v, Δ -> begin
        Δin = pb(Δ)
        g = ChainRulesCore.unthunk(Δin[2])
        if tf.on_grad
            grad = g isa Union{AbstractArray,Number} ? copy(g) : g
            # The `grad` field is a LIST of per-pass gradients, matching the
            # frule trace format: a later frule at the same `x` appends to it
            # (a raw gradient stored here would be corrupted by `push!`).
            push!(tf.trace, (input=copy(x), output=copy(v), grad=[grad]))
        end
        return (Δin[1], g)
    end
end
function ChainRulesCore.frule(rc::RuleConfig, Δs::Tuple, tf::TraceFunction, x)
    _, Δx = Δs
    v, g = ChainRulesCore.frule(rc, (NoTangent(), Δx), tf.f, x)

    if tf.on_grad
        if !isempty(tf.trace) &&
            x == tf.trace[end].input &&
            hasproperty(tf.trace[end], :grad)
            push!(tf.trace[end].grad, g)
        else
            push!(tf.trace, (input=copy(x), output=copy(v), grad=[copy(g)]))
        end
    end
    return v, g
end
function ChainRulesCore.frule(Δs::Tuple, tf::TraceFunction, x)
    _, Δx = Δs
    v, g = ChainRulesCore.frule((NoTangent(), Δx), tf.f, x)

    if tf.on_grad
        if !isempty(tf.trace) &&
            x == tf.trace[end].input &&
            hasproperty(tf.trace[end], :grad)
            push!(tf.trace[end].grad, g)
        else
            push!(tf.trace, (input=copy(x), output=copy(v), grad=[copy(g)]))
        end
    end
    return v, g
end

@ForwardDiff_frule (f::TraceFunction)(x::AbstractVector{<:ForwardDiff.Dual})
@ForwardDiff_frule (f::TraceFunction)(x::ForwardDiff.Dual)
