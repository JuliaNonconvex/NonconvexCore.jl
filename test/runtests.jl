using SafeTestsets, Test

@safetestset "Model" begin include("model.jl") end
@safetestset "DictModel" begin include("dict_model.jl") end
@safetestset "JuMP" begin include("jump.jl") end
@safetestset "Functions" begin include("functions.jl") end
