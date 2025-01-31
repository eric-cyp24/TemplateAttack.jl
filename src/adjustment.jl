

"""
function adjust!(t::Template, traces::AbstractMatrix; method=:emalg, 
                 num_epoch::Integer=200, δ=10e-9, dims=0, n_sample=0, Σscale=1)

`method`=:emalg or :normalize or :tr_normalize. 
`Σscale`= 0 -> only re-center the GMM (mvgs). 
`Σscale`= 1 -> normalize the mean and covMatrix of the GMM (mvgs). 
`Σscale`= 2 -> normalize GMM and expend each components' (mvg's) covMatrix. 
"""
function adjust!(t::Template, traces::AbstractMatrix; method=:emalg, 
                 num_epoch::Integer=200, δ=10e-9, dims=0, n_sample=0, Σscale=1)
    if method == :emalg
        adjust_emalg!(t, traces, num_epoch; δ, dims, n_sample, Σscale)
    elseif method == :normalize
        adjust_normalize!(t, traces; Σscale)
    elseif method == :tr_normalize
        traces[:] = trace_normalize(traces, t)
    end
end

function adjust_emalg!(t::Template, traces::AbstractMatrix, num_epoch::Integer=50; 
                       δ=10e-9, dims=0, n_sample=0, Σscale=0)
    dims   = dims==0 ? ndims(t) : dims
    n      = n_sample==0 ? (size(traces,2)÷length(t)÷4) : n_sample
    traces = ndims(t)==size(traces,1) ? traces[1:dims,:] : t.ProjMatrix[:,1:dims]' *traces

    # templates to GMM models
    t_new  = copy(t)
    templatedimreduce!( t_new,   dims)
    adjust_normalize!(  t_new, traces; Σscale) # 
    gmm, labels2gmmidx = templates2GMM(t_new; dims, n)
    print("              Number of GMM components: $(length(gmm))\r")
    
    # EM Algorithm
    llh = emalgorithm_fixedweight_mprocess!(gmm, traces, num_epoch; δ)

    # GMM back to templates
    # only modify the template  when EM algorithm succeed
    templatedimreduce!(t, dims)
    t.mean = dropdims(mean(traces,dims=2);dims=2)
    for (l,i) in labels2gmmidx
        μ, Σ = gmm.components[i].μ, gmm.components[i].Σ
        t.mvgs[l] = MvNormal(μ,Σ)
    end
    
    return t
end

"""
    adjust_normalize!(t::Template, traces::AbstractMatrix; Σscale=1)

`Σscale`= 0 -> only re-center the GMM (mvgs)
`Σscale`= 1 -> normalize the mean and covMatrix of the GMM (mvgs)
`Σscale`= 2 -> normalize GMM and expend each components' (mvg's) covMatrix
"""
function adjust_normalize!(t::Template, traces::AbstractMatrix; Σscale=1)
    traces  = ndims(t)==size(traces,1) ? traces : t.ProjMatrix' * traces
    tr_mean = dropdims(mean(traces,dims=2);dims=2)
    tr_std  = dropdims( std(traces,dims=2);dims=2)
    gmm     = GaussianMixtureModel(collect(values(t.mvgs)), collect(values(t.priors)))
    t_mean  = mean(gmm)
    t_std   = sqrt.(diag(cov(gmm)))
    scale   = Σscale==0 ? 1 : tr_std ./ t_std # in each LDA vector space
    Σscale  = Σscale==0 ? 1 : Σscale
    # modify the template
    for (l,mvg) in t.mvgs
        mu, sig   = (((mvg.μ - t_mean) .* scale) + tr_mean), Σscale*(mvg.Σ .* (scale*scale'))
        t.mvgs[l] = MvNormal(mu,sig)
    end
    return t
end

function trace_normalize!(traces::AbstractMatrix, t::Template)
    tr_mean, tr_var = vec(mean(traces,dims=2)), vec(var(traces,dims=2))
    return traces[:] = @. ((traces - tr_mean) * (sqrt.(t.TraceVar / tr_var))) + t.TraceMean
end
function trace_normalize(traces::AbstractMatrix, t::Template)
    return trace_normalize!(copy(traces),t)
end

### helper functions

function templates2GMM(t::Template; dims=0, n=50, pval=5e-3)
    dims    = dims==0 ? ndims(t) : dims
    labels  = sort(collect(keys(t.mvgs)))
    mvgs    = Vector{MvNormal}()
    weights = Vector{Float64}()
    labels2gmmidx = Dict()
    for l in labels
        mvg, p = mvgdimreduce(t.mvgs[l],dims), t.priors[l]
        merged = false
        # check if the mvg is the same as any of the privious one
        # point to the old one if match else, create a new mvg
        for (i,m) in enumerate(mvgs)
            merged = pvalue(UnequalCovHotellingT2Test(m, mvg, n)) > pval
            if merged
                weights[i] += p
                labels2gmmidx[l] = i
                break
            end
        end
        # add mvg model in to GMM component list
        if !merged
            push!(mvgs, mvg)
            push!(weights, p)
            labels2gmmidx[l] = length(mvgs)
        end
    end
    return GaussianMixtureModel(mvgs, weights), labels2gmmidx
end

function mvgdimreduce(mvg::MvNormal, dims)
    dims = min(length(mvg), dims)
    return MvNormal(mvg.μ[1:dims], mvg.Σ[1:dims,1:dims])
end

function templatedimreduce(t::Template, dims)
    return templatedimreduce!(copy(t), dims)
end
function templatedimreduce!(t::Template, dims)
    dims = min(ndims(t), dims)
    t.ProjMatrix     = t.ProjMatrix[:,1:dims]
    t.mean           = t.mean[1:dims]
    t.covMatrix      = t.covMatrix[1:dims,1:dims]
    t.pooled_cov_inv = t.pooled_cov_inv[1:dims,1:dims]
    for (l,mvg) in t.mvgs
        t.mvgs[l] = MvNormal(mvg.μ[1:dims], mvg.Σ[1:dims,1:dims])
    end
    return t
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


