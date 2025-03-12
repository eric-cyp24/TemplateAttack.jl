using Test
using TemplateAttack


@testset verbose=true "TemplateAttack.jl" begin
    @testset "template.jl" begin
        include("template_tests.jl")
    end

end
