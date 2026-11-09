"""
PlottingCallback

A callback function that plots a number of convergence metrics using [ConvergencePlots.jl](https://github.com/mohdibntarek/ConvergencePlots.jl). The following are the fields of the struct:
- `plot`: an instance of `ConvergencePlots.ConvergencePlot`.
- `n`: the number of history points to keep track of.
- `iter`: the number of times the callback was called.
"""
mutable struct PlottingCallback <: Function
    plot::Any
    n::Int
    iter::Int
end
PlottingCallback(n=1000) = PlottingCallback(nothing, n, 0)

(callback::PlottingCallback)(solution) = nothing

"""
    LazyPlottingCallback

A callback function that plots a number of convergence metrics using [ConvergencePlots.jl](https://github.com/mohdibntarek/ConvergencePlots.jl) only when `show` is called. The following are the fields of the struct:
- `plot`: an instance of `ConvergencePlots.ConvergencePlot`.
- `n`: the number of history points to keep track of.
- `iter`: the number of times the callback was called.
"""
mutable struct LazyPlottingCallback <: Function
    plot::Any
    n::Int
    iter::Int
    show_plot::Bool
    save_plot::Union{Nothing,String}
end
function LazyPlottingCallback(n=10^10; show_plot=true, save_plot=nothing)
    return LazyPlottingCallback(nothing, n, 0, show_plot, save_plot)
end

(callback::LazyPlottingCallback)(args...; kwargs...) = nothing

struct NoCallback <: Function end
(::NoCallback)(args...; kwargs...) = nothing
