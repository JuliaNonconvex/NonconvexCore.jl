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
