module NonconvexCoreConvergencePlotsExt

using NonconvexCore: NonconvexCore
using ConvergencePlots: ConvergencePlots
using NonconvexCore: PlottingCallback, LazyPlottingCallback, Solution
using ConvergencePlots: ConvergencePlot, addpoint!
using NonconvexCore: @unpack

"""
(callback::PlottingCallback)(solution::Solution)

Updates the convergence plots using the current solution.
"""
function (callback::PlottingCallback)(solution::Solution)
    @unpack convstate = solution
    if callback.iter == 0
        callback.plot = ConvergencePlot(
            callback.n,
            names=["KKT residual", "|Δf|", "|Δx|_∞", "Infeasibility"],
            #names = ["KKT residual", "IPOPT residual", "|Δf|", "|Δx|_∞", "Infeasibility"],
            options=Dict(
                "KKT residual" => (color="black",),
                #"IPOPT residual" => (color = "blue",),
                "|Δf|" => (color="green",),
                "|Δx|_∞" => (color="violet",),
                "Infeasibility" => (color="red",),
            ),
        )
    end
    @unpack plot = callback
    addpoint!(
        plot,
        Dict(
            "KKT residual" => convstate.kkt_residual,
            #"IPOPT residual" => convstate.ipopt_residual,
            "|Δf|" => convstate.Δf,
            "|Δx|_∞" => convstate.Δx,
            "Infeasibility" => convstate.infeas,
        ),
    )
    return callback.iter += 1
end

"""
(callback::LazyPlottingCallback)(solution::Solution)

Updates the convergence plots using the current solution.
"""
function (callback::LazyPlottingCallback)(solution::Solution; update=false)
    @unpack convstate = solution
    if callback.iter == 0
        callback.plot = ConvergencePlot(
            callback.n,
            names=["KKT residual", "|Δf|", "|Δx|_∞", "Infeasibility"],
            #names = ["KKT residual", "IPOPT residual", "|Δf|", "|Δx|_∞", "Infeasibility"],
            options=Dict(
                "KKT residual" => (color="black",),
                #"IPOPT residual" => (color = "blue",),
                "|Δf|" => (color="green",),
                "|Δx|_∞" => (color="violet",),
                "Infeasibility" => (color="red",),
            ),
            show=update && callback.show_plot,
        )
    end
    @unpack plot = callback
    addpoint!(
        plot,
        Dict(
            "KKT residual" => convstate.kkt_residual,
            #"IPOPT residual" => convstate.ipopt_residual,
            "|Δf|" => convstate.Δf,
            "|Δx|_∞" => convstate.Δx,
            "Infeasibility" => convstate.infeas,
        );
        show=callback.show_plot,
        update=update,
        filename=callback.save_plot,
    )
    return callback.iter += 1
end
end # module NonconvexCoreConvergencePlotsExt
