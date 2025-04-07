
using TemplateAttack:loaddata, writedata
@testset "test loaddata/writedata" begin
    A = rand(5,4); path = joinpath(@__DIR__, "test_loaddata.h5")
    writedata(path, A)
    A_load = loaddata(path)
    @test A_load == A && !(A_load === A)
    rm(path)
end

