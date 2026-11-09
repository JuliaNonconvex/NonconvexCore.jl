# Models

## `Model`

To define an empty model, run:

```julia
model = Model()
```

To specify an objective function `obj` when creating the model, run:

```julia
model = Model(obj)
```

where `obj` is a function that takes a single vector argument.

### Adding a single variable

To add a new variable to a `Model` with lower and upper bounds `lb` and `ub`
respectively, use:

```julia
addvar!(model, lb, ub)
```

The variables added will be stacked on top of each other with a linear
integer index. The lower and upper bounds for each variable don't have to be
numbers, they can be:

1. Dictionaries
2. Structs
3. `Tuple`s
4. `NamedTuple`s
5. Nested data structures

However, the values input cannot be vectors. A vector input has a different
interpretation; see the next section.

The types of the lower and upper bounds of each variable must be the same,
and this type is assumed to be the type of the decision variable. Different
variables can have different types. Vectorization and de-vectorization are
handled automatically.

To specify an initial value, use the `init` keyword argument. To add an
integer constraint on the variable, use the `integer` keyword argument. For
example:

```julia
addvar!(model, 0.0, 10.0, init = 1.0, integer = true)
```

`init` must have the same type as the lower and upper bounds.

### Adding multiple variables

To add multiple variables simultaneously, pass a vector of values for the
bounds and optionally for the `init` and `integer` keyword arguments:

```julia
addvar!(model, [0.0, 0.0], [10.0, 10.0], init = [1.0, 1.0], integer = [true, false])
```

The elements of the vector can be:

1. Vectors or arrays in general
2. Dictionaries
3. Structs
4. `Tuple`s
5. `NamedTuple`s
6. Nested data structures

Note that the use of vectors as elements is allowed here. Similarly, the
types of the lower and upper bounds and the initial values must be the same.

## `DictModel`

There are 2 ways to define a `DictModel`. The direct method is:

```julia
model = DictModel()
```

To pass an objective function while constructing the model, use:

```julia
model = DictModel(obj)
```

where `obj` is a function that takes a single `OrderedDict` argument.

### Adding a single variable

Each variable in a `DictModel` has a name, which can be a `Symbol` or a
`String`. Each named variable can have an arbitrary type, e.g:

1. Vectors or arrays in general
2. Dictionaries
3. Structs
4. `Tuple`s
5. `NamedTuple`s
6. Nested data structures

Vectorization and de-vectorization are handled automatically.

To add a new named variable to a `DictModel` with a name `:a` and lower and
upper bounds `lb` and `ub` respectively, use:

```julia
addvar!(model, :a, lb, ub)
```

Similar to `Model`, optional keyword arguments `init` and `integer` can be
set, and the types of the initial value, `lb` and `ub` must be the same.

### Adding multiple variables

There is no way to add multiple variables simultaneously to a `DictModel`,
however a single named variable that's a vector can be added.

## Objective definition

To specify an objective function after creating the model, use:

```julia
set_objective!(model, obj)
```

where `obj` is a function that takes the model's variable container: a single
vector for a `Model`, and a single `OrderedDict` for a `DictModel`. The input
to `obj` has the same structure, shape and types as the initial solution and
the bounds.

## Inequality constraint definition

To define an inequality constraint `f(x) <= 0`, where `f` is a Julia function
that accepts the model's variable container, use:

```julia
add_ineq_constraint!(model, f)
```

The function `f` can return:

1. A number, in which case the constraint is `f(x) <= 0`.
2. A vector or array of numbers, in which case the constraint is applied
   element-wise, `f(x) .<= 0`.
3. An arbitrary container or data structure, in which case the output is
   vectorized first and the constraint is applied element-wise on the
   vectorized output.

## Equality constraint definition

To define an equality constraint `f(x) == 0`, use:

```julia
add_eq_constraint!(model, f)
```

with the same conventions as inequality constraints.

## Semidefinite constraint definition

To define a semidefinite constraint, i.e. a symmetric matrix `f(x)` that must
be positive semidefinite, use:

```julia
add_sd_constraint!(model, f)
```

where `f` returns either a symmetric matrix or a vector of the
lower-triangular entries of the matrix.

## Changing variable bounds

After defining the variables, it is possible to set the minimum and maximum
variable bounds to different values. This is useful in iterative procedures
where the bounds are updated and the problem is resolved.

To update the entire vector of minimum bounds, use:

```julia
setmin!(model, newmin)
```

where `newmin` is a vector of bounds; `-Inf` is allowed. To set a new minimum
bound for the `i`th variable only, use:

```julia
setmin!(model, i, newmin)
```

Similarly, [`setmax!`](@ref) updates the maximum bounds, with `Inf` allowed.

## Changing integrality constraints

To constrain a variable to be integer or relax the integrality constraint on
the `i`th variable, use:

```julia
setinteger!(model, i, integer)
```

where `integer` is `true` to constrain the variable or `false` to relax the
constraint.
