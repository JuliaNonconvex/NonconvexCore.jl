# Docstrings

```@autodocs
Modules = [NonconvexCore]
Private = false
Order = [:module, :type, :function, :macro, :constant]
```
```@docs
NonconvexCore.AbstractFunction
NonconvexCore.AbstractConstraint
NonconvexCore.Objective
NonconvexCore.IneqConstraint
NonconvexCore.EqConstraint
NonconvexCore.SDConstraint
NonconvexCore.VectorOfFunctions
NonconvexCore.getfunction
NonconvexCore.getdim
NonconvexCore.value_gradient
NonconvexCore.value_jacobian
NonconvexCore.value_hessian
NonconvexCore.sparse_jacobian
NonconvexCore.sparse_hessian
NonconvexCore.second_order_backend
NonconvexCore.getinit
NonconvexCore.getnvars
NonconvexCore.getnconstraints
NonconvexCore.getineqconstraints
NonconvexCore.geteqconstraints
NonconvexCore.getobjectiveconstraints
NonconvexCore.getnineqconstraints
NonconvexCore.getneqconstraints
NonconvexCore.ConvergenceCriteria
NonconvexCore.ConvergenceState
NonconvexCore.assess_convergence!
NonconvexCore.JuMPEvaluator
NonconvexCore.JuMPModelObjective
NonconvexCore.JuMPModelIneq
NonconvexCore.JuMPModelEq
NonconvexCore.jump_objective_gradient
NonconvexCore.jump_jacobian
NonconvexCore.jump_jacobian_structure
NonconvexCore.jump_hessian_structure
```

```@docs
NonconvexCore.@params
NonconvexCore.get_variable_info
NonconvexCore._ordered_jump_variables
NonconvexCore._nonlinear_constraints
NonconvexCore.get_jump_problem
NonconvexCore.Solution
NonconvexCore.Trace
NonconvexCore.AbstractResult
NonconvexCore.GenericResult
NonconvexCore.PlottingCallback
NonconvexCore.LazyPlottingCallback
NonconvexCore.PreparedFunction
NonconvexCore.FlattenedFunction
NonconvexCore.DEFAULT_AD_BACKEND
NonconvexCore.check_finite_start!(::NonconvexCore.AbstractModel, ::Any)
Base.length(::NonconvexCore.AbstractFunction)
NonconvexCore.getinputunflatten(::NonconvexCore.FlattenedFunction)
NonconvexCore.getrawfunction(::NonconvexCore.FlattenedFunction)
NonconvexCore.getfunctions(::NonconvexCore.VectorOfFunctions)
NonconvexCore.set_objective(::NonconvexCore.AbstractModel, ::Function)
NonconvexCore.hasconverged(::NonconvexCore.Solution)
NonconvexCore.hasconverged(::NonconvexCore.ConvergenceState)
NonconvexCore.correctsolution!(::NonconvexCore.Solution, ::NonconvexCore.AbstractModel, ::Any)
NonconvexCore.clamp_and_evaluate!(::NonconvexCore.AbstractModel, ::AbstractVector)
NonconvexCore.jump_symbolic_hessian_sparsity
NonconvexCore.get_objective_multiple
NonconvexCore.set_objective_multiple!
NonconvexCore.tovecmodel(::NonconvexCore.DictModel, ::AbstractVector)
NonconvexCore.geteqconstraint
NonconvexCore.getineqconstraint
NonconvexCore.setbox!
NonconvexCore.sparse_gradient
```

```@docs
NonconvexCore.jump_constraint_linearity
```
