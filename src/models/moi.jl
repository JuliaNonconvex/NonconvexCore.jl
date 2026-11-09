# Utilities for sparse Jacobian/Hessian bookkeeping shared with the
# solver packages.
nvalues(::Nothing) = 0
nvalues(J::Matrix) = length(J)
nvalues(H::LowerTriangular{<:Real,<:Matrix}) = (size(H, 1) + 1) * size(H, 1) ÷ 2
nvalues(H::SparseMatrixCSC) = length(H.nzval)
function nvalues(HL::LowerTriangular{<:Real,<:SparseMatrixCSC})
    nvalues = 0
    H = HL.data
    for col = 1:length(H.colptr)-1
        indices = [i for i = H.colptr[col]:H.colptr[col+1]-1 if H.rowval[i] >= col]
        nvalues += length(indices)
    end
    return nvalues
end

# Implement these for sparse matrices
function fill_indices!(rows, cols, J0::Matrix; offset = 0, row_offset = 0)
    nconstr, nvars = size(J0)
    for j = 1:nvars
        cols[offset+1:offset+nconstr] .= j
        rows[offset+1:offset+nconstr] .= row_offset+1:row_offset+nconstr
        offset += nconstr
    end
    return rows, cols
end
function fill_indices!(
    rows,
    cols,
    HL::LowerTriangular{<:Real,<:Matrix};
    offset = 0,
    row_offset = 0,
)
    nvars = size(HL, 1)
    for j = 1:nvars
        cols[offset+1:offset+nvars-j+1] .= j
        rows[offset+1:offset+nvars-j+1] .= row_offset+j:row_offset+nvars
        offset += nvars - j + 1
    end
    return rows, cols
end
function fill_indices!(rows, cols, HL::SparseMatrixCSC; offset = 0, row_offset = 0)
    for col = 1:length(HL.colptr)-1
        indices = HL.colptr[col]:HL.colptr[col+1]-1
        nvars = length(indices)
        cols[offset+1:offset+nvars] .= col
        rows[offset+1:offset+nvars] = row_offset .+ HL.rowval[indices]
        offset += nvars
    end
    return rows, cols
end
function fill_indices!(
    rows,
    cols,
    HL::LowerTriangular{<:Real,<:SparseMatrixCSC};
    offset = 0,
    row_offset = 0,
)
    H = HL.data
    for col = 1:length(H.colptr)-1
        indices = [i for i = H.colptr[col]:H.colptr[col+1]-1 if H.rowval[i] >= col]
        nvars = length(indices)
        cols[offset+1:offset+nvars] .= col
        rows[offset+1:offset+nvars] = row_offset .+ H.rowval[indices]
        offset += nvars
    end
    return rows, cols
end

function add_values!(values, J::Matrix; offset = 0)
    nvars = length(J)
    values[offset+1:offset+nvars] .+= vec(J)
    return values
end
function add_values!(values, HL::LowerTriangular{<:Real,<:Matrix}; factor = 1, offset = 0)
    nvars = size(HL, 1)
    for j = 1:nvars
        values[offset+1:offset+nvars-j+1] .+= HL[j:nvars, j] .* factor
        offset += nvars - j + 1
    end
    return values
end
function add_values!(values, HL::SparseMatrixCSC; factor = 1, offset = 0)
    nvars = length(HL.nzval)
    values[offset+1:offset+nvars] .= HL.nzval .* factor
    return values
end
function add_values!(
    values,
    HL::LowerTriangular{<:Real,<:SparseMatrixCSC};
    factor = 1,
    offset = 0,
)
    H = HL.data
    for col = 1:length(H.colptr)-1
        indices = [i for i = H.colptr[col]:H.colptr[col+1]-1 if H.rowval[i] >= col]
        nvars = length(indices)
        values[offset+1:offset+nvars] .= H.nzval[indices] .* factor
        offset += nvars
    end
    return values
end

_dot(f, x, y) = dot(f(x), y)
_dot(::Nothing, ::Any, ::Any) = 0.0
function _dot(f::VectorOfFunctions, x, y)
    # Accumulate one term per function instead of calling `f(x)` as a whole so
    # that reverse-mode backends differentiate each function separately.
    s = 0.0
    offset = 0
    for fi in f.fs
        dim = getdim(fi)
        s += dot(fi(x), @view(y[offset+1:offset+dim]))
        offset += dim
    end
    return s
end

struct JuMPProblem{E,M,V}
    evaluator::E
    model::M
    vars::V
end

struct JuMPProblemEvaluator{XL,XU,X,CL,CU,O,C,G,J,JS,H,HS} <: MOI.AbstractNLPEvaluator
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

