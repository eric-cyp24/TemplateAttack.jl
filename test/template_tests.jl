using Test
using TemplateAttack

@testset "Template load/dump" begin
    t_ref = loadtemplate(joinpath(@__DIR__, "template_ref.h5"))
    writetemplate(joinpath(@__DIR__, "template_test.h5"),t_ref)
    writetemplate(joinpath(@__DIR__, "template_test.h5"),t_ref;byte=1)
    t0,t1 = [loadtemplate(joinpath(@__DIR__, "template_test.h5");byte=i) for i in 0:1]
    @test (t_ref == t0 == t1) && (false == (t0 === t1 || t_ref === t1 || t_ref === t1))
    rm(joinpath(@__DIR__, "template_test.h5"))
end
