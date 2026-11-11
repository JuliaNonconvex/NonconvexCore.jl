# Gradients, Jacobians and Hessians

By default, `NonconvexCore` (and hence `Nonconvex`, which re-exports it)
computes gradients and Jacobians using the reverse-mode automatic
differentiation (AD) package, [`Zygote.jl`](https://github.com/FluxML/Zygote.jl),
through the [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
backend system; see [AD backends](../derivatives/backends.md).

However, one can force `NonconvexCore` to use other AD packages or even user-defined gradients and Hessians using special function modifiers. Those special function modifiers customize the behaviour of functions without enforcing the same behaviour on other functions. For instance:
- A specific AD package can be used for one constraint function while the default AD packages are used for other functions in the optimization problem.
- The history of gradients of a specific function can be stored without storing all the gradients of all the functions.
- For functions with a sparse Jacobian or Hessian, the sparsity can be used to speedup the AD using sparse, forward-mode AD for these functions.

In some cases, function modifiers can even be composed on top of each other to create more complex behaviours. 

> In `NonconvexCore`, function modifiers modify the behaviour of a function when differentiated once or twice using either `ForwardDiff` or any [`ChainRules`](https://github.com/JuliaDiff/ChainRules.jl)-compatible AD package, such as `Zygote.jl`. All of the following features are implemented in `NonconvexCore.jl` and re-exported from `Nonconvex`.

## Table of contents

```@contents
Pages = ["user_defined.md", "chainrules_fd.md", "history.md"]
Depth = 3
```
