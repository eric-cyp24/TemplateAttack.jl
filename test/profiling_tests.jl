

@testset "linear discriminant analysis (LDA)" begin
    for i in 1:5
        # gen data
        N  = 20000; A = 10*rand(2,2) .+ [1 0;0 1]
        g1 = [1.0, -0.5]; g2 = [-1.0, 0.5]; noise = [1, 0.1]
        data  = A * [(g1 .+(randn(2,N).*noise))  (g2 .+(randn(2,N).*noise))]
        label = vcat(ones(Int8,N),2*ones(Int8,N))
        # compute LDA for the gen data
        SB = cov(A * [g1 g2]; dims=2, corrected=false)
        SW = A * diagm(noise.^2) * A'
        V  = normalize(eigen(SB,SW).vectors[:,2])
        # test LDA implementation
        U  = normalize(LDA(data, label; outdim=1))
        @test U[1]/U[2] ≈ V[1]/V[2]  rtol=0.01 # compare LDA dirctions

        ## show in plot
        # plotdatascatter(data;groupdict=groupbyval(label),aspect_ratio=:equal)
        # scale = maximum(abs.(data))/2
        # plot!(eachrow(scale*[U -U])...; label="LDA vector")
        # plot!(eachrow(scale*[V -V])...; label="LDA compute")
    end
end

include("leakagesim.jl")
N = 25600; nbytes = 4; value_range = 0:7
IVs, traces = gendataset(N, nbytes; value_range, noise=0.05)

using TemplateAttack:findPOI, expandPOI
@testset "findPOI" begin
    for i in 1:nbytes
        poi = findPOI(IVs[i,:], traces; nicv_th=0.2)
        @test poi == collect((i*4-3:i*4).+1)
    end
end

@testset "expandPOI" begin
    poi = [2,10,14]
    @test expandPOI(poi,18,0,0) == [2,10,14]
    @test expandPOI(poi,18,1,0) == [1,2,9,10,13,14]
    @test expandPOI(poi,18,2,0) == [1,2,8,9,10,12,13,14]
    @test expandPOI(poi,18,2,3) == [1,2,3,4,5,8,9,10,11,12,13,14,15,16,17]
    @test expandPOI(poi,18,2,6) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    @test expandPOI(poi,18,0,6) == [2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18]
end

@testset "buildTemplate" begin
    cyclemean = mean(count_ones.(value_range))*wavecycle
    wavemean = zeros(4*(size(IVs,1)+1))
    for i in 1:size(IVs,1) wavemean[i*4-3:i*4+4] .+= cyclemean end
    for byte in 1:nbytes
        t = buildTemplate(IVs[byte,:], traces; nicv_th=0.1, numofcomponents=1)
        @test t.TraceMean ≈ wavemean                    rtol=0.01
        @test t.mean      ≈ t.ProjMatrix' *wavemean     rtol=0.01
        for val in value_range
            exptrace = deepcopy(wavemean)
            exptrace[byte*4-3:byte*4+4] += (count_ones(val)*wavecycle -cyclemean)
            @test t.ProjMatrix' *exptrace ≈ t.mvgs[val].μ   rtol=0.01 atol=0.005
        end
    end
end
