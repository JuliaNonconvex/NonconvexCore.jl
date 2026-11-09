using SafeTestsets, Test

@safetestset "Model" begin
    include("model.jl")
end
@safetestset "DictModel" begin
    include("dict_model.jl")
end
@safetestset "JuMP" begin
    include("jump.jl")
end
@safetestset "MOI" begin
    include("moi.jl")
end
@safetestset "AD backends" begin
    include("ad_backends.jl")
end
@safetestset "Functions" begin
    include("functions.jl")
end
@safetestset "Convergence" begin
    include("convergence.jl")
end
@safetestset "Capabilities" begin
    include("capabilities.jl")
end
@safetestset "Regressions" begin
    include("regressions.jl")
end

@safetestset "ForwardDiff frule" begin
    include("forwarddiff_frule.jl")
end
@safetestset "TraceFunction" begin
    include("trace.jl")
end
@safetestset "Custom gradients" begin
    include("custom.jl")
end
@safetestset "Symbolics" begin
    include("symbolic.jl")
end
@safetestset "Coverage" begin
    include("coverage.jl")
end
