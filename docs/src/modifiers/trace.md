# Tracing calls

A [`TraceFunction`](@ref) wraps a function and records information about
where it is evaluated, which is useful for debugging which part of a model
calls a function during optimization.

```julia
f = TraceFunction(sum; on_call = true, on_grad = true)
```

The recorded trace can be inspected after evaluations have happened. See the
[`TraceFunction`](@ref) docstring for the details of what is recorded and the
available options.
