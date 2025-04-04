
using TemplateAttack:loaddata, writedata
@testset "test loaddata/writedata" begin
    A, path = rand(5,4), "test_loaddata.h5"
    writedata(path, A)
    A_load = loaddata(path)
    @test A_load == A && !(A_load === A)
    rm(path)
end

