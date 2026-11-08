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
