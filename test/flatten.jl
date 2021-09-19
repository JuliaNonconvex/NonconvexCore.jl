using NonconvexCore: flatten
using OrderedCollections, JuMP, Zygote, SparseArrays, LinearAlgebra

struct SS
    a
    b
end

xs = [
    1.0,
    [1.0],
    [1.0, 2.0],
    [1.0, [1.0, 2.0]],
    [1.0, (1.0, 2.0)],
    [1.0, OrderedDict(1 => [1.0, 2.0])],
    [[1.0], OrderedDict(1 => [1.0, 2.0])],
    [(1.0,), [1.0,], OrderedDict(1 => [1.0, 2.0])],
    [1.0 1.0; 1.0 1.0],
    rand(2, 2, 2),
    [[1.0, 2.0], [3.0, 4.0]],
    OrderedDict(1 => 1.0),
    OrderedDict(1 => [1.0]),
    OrderedDict(1 => 1.0, 2 => [2.0]),
    OrderedDict(1 => 1.0, 2 => [2.0], 3 => [[1.0, 2.0], [3.0, 4.0]]),
    JuMP.Containers.DenseAxisArray(reshape([1.0, 1.0], (2,)), 1),
    (1.0,),
    (1.0, 2.0),
    (1.0, (1.0, 2.0)),
    (1.0, [1.0, 2.0]),
    (1.0, OrderedDict(1 => [1.0, 2.0])),
    ([1.0], OrderedDict(1 => [1.0, 2.0])),
    ((1.0,), [1.0,], OrderedDict(1 => [1.0, 2.0])),
    (a = 1.0,),
    (a = 1.0, b = 2.0),
    (a = 1.0, b = (1.0, 2.0)),
    (a = 1.0, b = [1.0, 2.0]),
    (a = 1.0, b = OrderedDict(1 => [1.0, 2.0])),
    (a = [1.0], b = OrderedDict(1 => [1.0, 2.0])),
    (a = (1.0,), b = [1.0,], c = OrderedDict(1 => [1.0, 2.0])),
    sparsevec([1.0, 2.0], [1, 3], 10),
    sparse([1, 2, 2, 3], [2, 3, 1, 4], [1.0, 2.0, 3.0, 4.0], 10, 10),
    SS(1.0, 2.0),
    [SS(1.0, 2.0), 1.0],
]

for x in xs
    @show x
    xvec, unflatten = flatten(x)
    @test x == unflatten(xvec)
    J = Zygote.jacobian(xvec) do x
        unflatten(x)
        flatten(x)[1]
    end[1]
    @test logabsdet(J) == (0.0, 1.0)
end
