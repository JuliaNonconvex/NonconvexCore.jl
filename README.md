<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexCore.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexCore.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexCore.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexCore.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexCore.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexCore.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexCore.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexCore.jl

The foundation of the JuliaNonconvex ecosystem

Models, automatic differentiation, function modifiers, solver capabilities and convergence criteria shared by every solver package.

NonconvexCore.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
using NonconvexCore

m = Model(x -> sum(x .^ 2))
addvar!(m, [-5.0, -5.0], [5.0, 5.0])
r = optimize(m, NelderMeadAlg(), [1.0, 1.0])
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexCore.jl/stable](https://julianonconvex.github.io/NonconvexCore.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexCore.jl/dev](https://julianonconvex.github.io/NonconvexCore.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexCore")
```
