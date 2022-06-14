# workaround Zygote.jacobian and ForwardDiff.jacobian not supporting sparse jacobians

function sparse_jacobian(f, x)
    val, pb = Zygote.pullback(f, x)
    M, N = length(val), length(x)
    T = eltype(val)
    return copy(mapreduce(hcat, 1:M, init = spzeros(T, N, 0)) do i
        pb(I(M)[:, i])[1]
    end')
end
function sparse_hessian(f, x)
    return sparse_fd_jacobian(x -> Zygote.gradient(f, x)[1], x)
end

function sparse_fd_jacobian(f, x)
    pf = pushforward_function(f, x)
    M, N = length(f(x)), length(x)
    init = pf(I(M)[:, 1])[1]
    M = length(init)
    return mapreduce(hcat, 2:N; init) do i
        pf(I(M)[:, i])[1]
    end
end

# from AbstractDifferentiation
function pushforward_function(f, xs...)
    return function pushforward(vs)
        if length(xs) == 1
            v = vs isa Tuple ? only(vs) : vs
            return (ForwardDiff.derivative(h -> f(step_toward(xs[1], v, h)), 0),)
        else
            return ForwardDiff.derivative(h -> f(step_toward.(xs, vs, h)...), 0)
        end
    end
end

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v
