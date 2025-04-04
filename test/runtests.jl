using Test
using Statistics, LinearAlgebra
using TemplateAttack


@testset verbose=true "TemplateAttack.jl" begin
    @testset verbose=true "template.jl" begin
        include("template_tests.jl")
    end
    @testset verbose=true "utils.jl" begin
        include("utils_tests.jl")
    end
    @testset verbose=true "profiling.jl" begin
        include("profiling_tests.jl")
    end

end
