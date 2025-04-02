using Test
using Statistics, LinearAlgebra
using TemplateAttack


@testset verbose=true "TemplateAttack.jl" begin
    @testset "template.jl" begin
        include("template_tests.jl")
    end

    @testset "profiling.jl" begin
        include("profiling_tests.jl")
    end

end
