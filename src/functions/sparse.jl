# Sparse Jacobians and Hessians via automatic sparse differentiation.
# SparseConnectivityTracer detects the structural sparsity pattern,
# SparseMatrixColorings compresses it into a graph coloring and
# DifferentiationInterface evaluates the compressed derivatives with the dense
# `backend`. If the sparse pipeline fails for a function, the derivatives are
# assembled from per-function derivatives at the detected pattern positions.

function _sparse_backend(backend::AbstractADType)
    return AutoSparse(backend, TracerSparsityDetector(), GreedyColoringAlgorithm())
end

"""
    second_order_backend(backend::ADTypes.AbstractADType)

Returns the backend used to compose second derivatives from `backend` for
`sparse_hessian`. The default is forward-mode differentiation over the
gradient computed with `backend`
(`DI.SecondOrder(AutoForwardDiff(), backend)`). This function can be
specialized for specific backends, e.g. to use a forward-over-reverse
composition with the same engine.
"""
second_order_backend(backend::AbstractADType) = DI.SecondOrder(AutoForwardDiff(), backend)

# Prepared sparse differentiation caches. Preparation performs the sparsity
# detection and graph coloring once; later evaluations reuse them. The
# functions stored in a VecModel are immutable, so caching by function
# identity is sound. The Hessian cache is keyed by identity as well: solvers
# pass a stable Lagrangian object (e.g. a closure over mutable multipliers).
const _SPARSE_JACOBIAN_PREP_CACHE = IdDict{Any,Any}()
const _SPARSE_HESSIAN_PREP_CACHE = IdDict{Any,Any}()

"""
    sparse_gradient(f, x, [backend::ADTypes.AbstractADType])

Returns the gradient of the scalar-valued function `f` at `x` computed with the
automatic differentiation backend `backend` (defaults to the package-wide
backend, see [`get_ad_backend`](@ref)).
"""
function sparse_gradient(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    _, g = DI.value_and_gradient(f, backend, x)
    return _sparsevec(g)
end

"""
    sparse_jacobian(f, x, [backend::ADTypes.AbstractADType])

Returns the structurally sparse Jacobian of `f` at `x` as a
`SparseMatrixCSC`. The sparsity pattern is detected with
SparseConnectivityTracer, colored with SparseMatrixColorings and the entries
are computed with the automatic differentiation backend `backend` (defaults
to the package-wide backend, see [`get_ad_backend`](@ref)). If the sparse
pipeline fails for `f`, the values are assembled from per-function
derivatives at the detected pattern positions. Failures (for example,
sparsity detection on functions whose internals branch on values) raise
errors rather than degrading silently.
"""
function sparse_jacobian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    return _sparse_jacobian_default(f, x, backend)
end

function _sparse_jacobian_default(f, x, backend::AbstractADType)
    y = f(x)
    # The sparse pipeline requires an array-valued function; promote scalar
    # outputs to 1-element vectors.
    f_ = y isa Number ? (z -> [f(z)]) : f
    # Reverse-mode differentiation of a whole collection through Zygote can
    # produce corrupted cotangents when the functions reconstruct flattened
    # structures; evaluate each function separately in that case.
    if backend isa AutoZygote
        pattern = jacobian_sparsity(f_, x, TracerSparsityDetector())
        return _assemble_sparse_jacobian(f_, x, pattern, backend)
    end
    key = (f, backend)
    entry = get(_SPARSE_JACOBIAN_PREP_CACHE, key, nothing)
    if entry === nothing
        sb = _sparse_backend(backend)
        entry = (f_, sb, DI.prepare_jacobian(f_, sb, x))
        _SPARSE_JACOBIAN_PREP_CACHE[key] = entry
    end
    f2, sb, prep = entry
    return DI.jacobian(f2, prep, sb, x)
end


function _assemble_sparse_jacobian(f, x, pattern, backend)
    m, n = size(pattern)
    I = Int[]
    J = Int[]
    V = Float64[]
    blocks = if f isa VectorOfFunctions
        map(fi -> Matrix(value_jacobian(fi, x, backend)[2]), f.fs)
    else
        [Matrix(value_jacobian(f, x, backend)[2])]
    end
    offset = 0
    for bi in blocks
        dim = size(bi, 1)
        sub = pattern[(offset+1):(offset+dim), :]
        for (lr, c) in zip(findnz(sub)[1], findnz(sub)[2])
            push!(I, offset + lr)
            push!(J, c)
            push!(V, bi[lr, c])
        end
        offset += dim
    end
    return sparse(I, J, V, m, n)
end

"""
    sparse_hessian(f, x, [backend::ADTypes.AbstractADType])

Returns the structurally sparse Hessian of the scalar-valued function `f` at
`x` as a `SparseMatrixCSC`. The sparsity pattern is detected with
SparseConnectivityTracer, colored with SparseMatrixColorings and the entries
are computed by forward-mode differentiation of the gradient computed with
`backend` (defaults to the package-wide backend, see [`get_ad_backend`](@ref)).
Failures raise errors rather than degrading silently.
"""
function sparse_hessian(f, x, backend::AbstractADType = DEFAULT_AD_BACKEND[])
    key = (f, backend)
    entry = get(_SPARSE_HESSIAN_PREP_CACHE, key, nothing)
    if entry === nothing
        sb = _sparse_backend(second_order_backend(backend))
        entry = (sb, DI.prepare_hessian(f, sb, x))
        _SPARSE_HESSIAN_PREP_CACHE[key] = entry
    end
    sb, prep = entry
    return DI.hessian(f, prep, sb, x)
end

_sparsevec(x::Real) = [x]
_sparsevec(x::Vector) = copy(x)
_sparsevec(x::Matrix) = copy(vec(x))
_sparsevec(x::SparseVector) = x
function _sparsevec(x::Adjoint{<:Real,<:AbstractMatrix})
    return _sparsevec(copy(x))
end
function _sparsevec(x::SparseMatrixCSC)
    m, n = size(x)
    linear_inds = zeros(Int, length(x.nzval))
    count = 1
    for colind = 1:length(x.colptr)-1
        for ind = x.colptr[colind]:x.colptr[colind+1]-1
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
    for (i, ind) in enumerate(v.nzind)
        _col, _row = divrem(ind, m)
        if _row == 0
            col = _col
            row = m
        else
            col = _col + 1
            row = _row
        end
        I[i] = row
        J[i] = col
    end
    return sparse(I, J, copy(v.nzval), m, n)
end

function ChainRulesCore.rrule(::typeof(_sparsevec), x)
    val = _sparsevec(x)
    val, Δ -> (NoTangent(), _sparse_reshape(Δ, size(x)...))
end
