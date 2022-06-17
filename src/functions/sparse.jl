# workaround Zygote.jacobian and ForwardDiff.jacobian not supporting sparse jacobians
function sparse_gradient(f, x)
    _, pb = Zygote.pullback(f, x)
    return _sparsevec(pb(1.0)[1])
end
function sparse_jacobian(f, x)
    val, pb = Zygote.pullback(f, x)
    M = length(val)
    vecs = [sparsevec([i], [true], M) for i in 1:M]
    Jt = reduce(hcat, first.(pb.(vecs)))
    return copy(Jt')
end
function sparse_fd_jacobian(f, x)
    pf = pushforward_function(f, x)
    M = length(x)
    vecs = [sparsevec([i], [true], M) for i in 1:M]
    # assumes there will be no structural non-zeros that have value 0
    return reduce(hcat, sparse.(pf.(vecs)))
end
function pushforward_function(f, x)
    return v -> begin
        return ForwardDiff.derivative(h -> f(x + h * v), 0)
    end
end

function sparse_hessian(f, x)
    return sparse_fd_jacobian(x -> sparse_gradient(f, x), x)
end

_sparsevec(x::Real) = [x]
_sparsevec(x::Vector) = copy(x)
_sparsevec(x::Matrix) = copy(vec(x))
_sparsevec(x::SparseVector) = x
function _sparsevec(x::Adjoint{<:Real, <:AbstractMatrix})
    return _sparsevec(copy(x))
end
function _sparsevec(x::SparseMatrixCSC)
    m, n = size(x)
    linear_inds = zeros(Int, length(x.nzval))
    count = 1
    for colind in 1:length(x.colptr)-1
        for ind in x.colptr[colind]:x.colptr[colind+1]-1
            rowind = x.rowval[ind]
            val = x.nzval[ind]
            linear_inds[count] = rowind + (colind - 1) * m
            count += 1
        end
    end
    return sparsevec(linear_inds, copy(x.nzval), prod(size(x)))
end

_sparse_reshape(v::AbstractVector, _) = v
_sparse_reshape(v::Vector, m, n) = reshape(v, m, n)
function _sparse_reshape(v::SparseVector, m, n)
    if length(v.nzval) == 0
        return sparse(Int[], Int[], v.nzval, m, n)
    end
    N = length(v.nzval)
    I = zeros(Int, N)
    J = zeros(Int, N)
    for ind in v.nzind
        _col, _row = divrem(ind, m)
        if _row == 0
            col = _col
            row = m
        else
            col = _col + 1
            row = _row
        end
        I[ind] = row
        J[ind] = col
    end
    return sparse(I, J, copy(v.nzval), m, n)
end

function ChainRulesCore.rrule(::typeof(_sparsevec), x)
    val = _sparsevec(x)
    val, Δ -> (NoTangent(), _sparse_reshape(Δ, size(x)...))
end
