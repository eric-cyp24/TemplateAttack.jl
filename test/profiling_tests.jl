

@testset "linear discriminant analysis (LDA)" begin
    for i in 1:5
        # gen data
        N, A = 10000, rand(2,2)
        g1, g2, noise = [1.0, -0.5], [-1.0, 0.5], [1, 0.1]
        data  = A*[g1.+(randn(2,N).*noise) g2.+(randn(2,N).*noise)]
        label = vcat(ones(Int8,N),2*ones(Int8,N))
        # compute LDA for the gen data
        SB = cov(A*[g1 g2]; dims=2, corrected=false)
        SW = A*diagm(noise.^2)*A'
        V  = eigen(SB,SW).vectors[:,2]
        # test LDA implementation
        U = LDA(data, label; outdim=1)
        @test U[1]/U[2] ≈ V[1]/V[2]  rtol=0.01 # compare LDA dirctions
        
        ## show in plot
        # plotdatascatter(data;groupdict=groupbyval(label),aspect_ratio=:equal)
        # plot!(eachrow( 10*[U -U])...; label="LDA vector")
        # plot!(eachrow(0.1*[V -V])...; label="LDA compute")
    end
end

