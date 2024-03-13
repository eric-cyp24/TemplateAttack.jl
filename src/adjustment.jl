using  LinearAlgebra, Statistics
import HypothesisTests: pvalue, UnequalCovHotellingT2Test, At_Binv_A
import Npy: loadnpy, isnpy
using  EMAlgorithm: GaussianMixtureModel, emalgorithm_fixedweight_mprocess!

function adjust!(t::Template, traces::AbstractMatrix; method=:emalg, 
                 num_epoch::Integer=200, δ=10e-9, dims=nothing, n_sample=nothing)
    if method == :emalg
        adjust_emalg!(t, traces, num_epoch; δ, dims, n_sample)
    elseif method == :normalize
        adjust_normalize!(t, traces)
    elseif method == :tr_normalize
        traces[:] = trace_normalize(traces, t)
    end
end

function adjust_emalg!(t::Template, traces::AbstractMatrix, num_epoch::Integer=50; 
                       δ=10e-9, dims=nothing, n_sample=nothing)
    traces = ndims(t)==size(traces,1) ? traces : t.ProjMatrix' * traces

    # templates to GMM models
    dims = isnothing(dims) ? size(traces,1) : dims
    n    = isnothing(n_sample) ? (size(traces,2)÷length(t)÷2) : n_sample
    gmm, labels2gmmidx = templates2GMM(t;dims,n)
    gmm = gmm + vec(mean(traces,dims=2) - t.mean)
    
    # EM Algorithm
    llh = emalgorithm_fixedweight_mprocess!(gmm, traces, num_epoch; δ)

    # GMM back to templates
    for (l,i) in labels2gmmidx
        μ, Σ = gmm.components[i].μ, gmm.components[i].Σ
        t.mvgs[l] = MvNormal(μ,Σ)
    end
    
    return t
end

function adjust_normalize!(t::Template, traces::AbstractMatrix)
    traces = ndims(t)==size(traces,1) ? traces : t.ProjMatrix' * traces
    tr_mean, scale = vec(mean(traces,dims=2)), sqrt.(vec(var(traces,dims=2))./diag(t.covMatrix))
    for (l,mvg) in t.mvgs
        mu = ((mvg.μ - t.mean) .* scale) + tr_mean
        t.mvgs[l] = MvNormal(mu,t.mvgs[l].Σ)
    end
end

function trace_normalize(traces::AbstractMatrix, t::Template)
    tr_mean, tr_var = vec(mean(traces,dims=2)), vec(var(traces,dims=2))
    return ((traces .- tr_mean) .* (sqrt.(t.TraceVar ./ tr_var))) .+ t.TraceMean
end

### helper functions

function templates2GMM(t::Template; dims=8, n=50, pval=5e-5)
    labels  = sort(collect(keys(t.mvgs)))
    mvgs    = Vector{MvNormal}()
    weights = Vector{Float64}()
    labels2gmmidx = Dict()
    for l in labels
        mvg,p = mvgdimreduce(t.mvgs[l],dims), t.priors[l]
        # check if the mvg is the same as any of the privious one
        # point to the old one if match else, create a new mvg
        for (i,m) in enumerate(mvgs)
            m = mvgdimreduce(m,dims)
            mvg_nodiff = pvalue(UnequalCovHotellingT2Test(m, mvg, n)) > pval
            if m.μ == mvg.μ || mvg_nodiff
                weights[i] += p
                mvg = nothing
                labels2gmmidx[l] = i
                break
            end
        end
        # add mvg model in to GMM component list
        if !isnothing(mvg)
            push!(mvgs,t.mvgs[l])
            push!(weights,p)
            labels2gmmidx[l] = length(mvgs)
        end
    end
    println("Number of GMM components: $(length(mvgs))")
    return GaussianMixtureModel(mvgs, weights), labels2gmmidx
end

function mvgdimreduce(mvg::MvNormal,dims)
    dims = min(length(mvg),dims)
    return MvNormal(mvg.μ[1:dims],mvg.Σ[1:dims,1:dims])
end

function mvgdistance(t1::MvNormal, t2::MvNormal)
    #mahalanobis(x, y, Q) = sqrt((x - y)' * Q * (x - y))
    return sqrt((t1.μ - t2.μ)' * t1.Σ * (t1.μ - t2.μ))
end

function UnequalCovHotellingT2Test(X::MvNormal, Y::MvNormal, n=50)
    p, nx, ny, Δ = length(X.μ), n, n, vec(X.μ - Y.μ)
    Sx, Sy, ST = X.Σ/nx , Y.Σ/ny, X.Σ/nx.+Y.Σ/ny
    T2 = At_Binv_A(Δ, ST)
    F = (nx + ny - p - 1) * T2 / (p * (nx + ny - 2))
    tmp = Symmetric(ST) \ Δ
    iv = (dot(tmp, Sx * tmp) / T2)^2 / (nx - 1) + (dot(tmp, Sy * tmp) / T2)^2 / (ny - 1)
    v = trunc(Int, inv(iv))
    return UnequalCovHotellingT2Test(T2, F, nx, ny, p, v, Δ, ST)
end


##### testing ground #####














