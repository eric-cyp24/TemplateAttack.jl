

### Template profiling

"""
    findPOI(IVs, traces; nicv_th=nothing, traceavg=nothing)

use NICV to find POIs, by setting a NICV threshold.
"""
function findPOI(IVs::AbstractVector, traces; nicv_th=nothing, traceavg=nothing)
    traceavg = isnothing(traceavg) ? view(traces,:,1:50) : traceavg
    print("\rCalculating NICV...\e[K\r")
    nicv = NICV(IVs, traces) # TODO: use lower level NICV calculation
    if isnothing(nicv_th)
        plotNICV(nicv, traceavg)
        print("Please Enter the NICV threshold: ")
        nicv_th = parse(Float64, readline())
    end
    return findall(>(nicv_th),nicv)
end

"""
    expandPOI(pois, trlen, POIe_left, POIe_right)

Given selected POIs, expend to neighbouring regions
"""
function expandPOI2(pois::AbstractVector, trlen, POIe_left, POIe_right)
    pois_set = Set(pois)
    for i in pois
        for p in i-POIe_left:i+POIe_right
            0 < p ≤ trlen && push!(pois_set, p)
        end
    end
    return sort!(collect(pois_set))
end

function expandPOI(pois::AbstractVector, trlen, POIe_left, POIe_right)
    (p,state) = iterate(sort(pois))
    poie = typeof(pois)(undef, 0);
    i = 1
    while i <= trlen
        if i < p-POIe_left i = p-POIe_left end
        append!(poie,i:min(trlen,p+POIe_right))
        i = poie[end]+1
        next = iterate(pois,state)
        if isnothing(next)
            return poie
        else
            (p,state) = next
        end
    end
    return poie
end

function compresstraces(traces, pois::AbstractVector; memmap::Bool=true, TMPFILE::AbstractString=TMPFILE)
    if memmap
        print("\rwriting to tmp...\e[K\r")
        fname, f = isdir(dirname(TMPFILE)) ? (TMPFILE, open(TMPFILE, "w+")) : mktemp()
        dtype, dsize = typeof(traces), (length(pois),size(traces,2))
        traces_poi   = mmap(f, dtype, dsize)
        traces_poi  .= view(traces,pois,:)
        close(f)
        return open(fname) do f mmap(f, dtype, dsize) end
    else
        return traces[pois,:]
    end
end

function LDA_projection_matrix(traces::Matrix{TT}, grouplist, numofcomponents=0) where {TT}
    info(msg) = print("\rLinear Discriminant Analysis...   ",msg,"\e[K")
    TracesMean = vec(mean(traces,dims=2))
    TraceMeans = stack([vec(mean(view(traces,:,gl), dims=2)) for gl in grouplist])

    info("finding Sb...")
    SB = zeros(TT,size(traces,1),size(traces,1))
    for (gl,t) in zip(grouplist, eachcol(TraceMeans.-TracesMean))
        LinearAlgebra.BLAS.syr!('U',TT(length(gl)),t,SB)
    end
    SB = Symmetric(SB) # syrk!('U',...) return only upper triangle

    info("finding Sw...")
    SW = zeros(TT,size(traces,1),size(traces,1))
    for (i,gl) in enumerate(grouplist)
        T = view(traces,:,gl) .- view(TraceMeans,:,i)
        LinearAlgebra.BLAS.syrk!('U','N',true,T,true,SW)
    end
    SW = Symmetric(SW)
    GC.gc(); # release memory

    info("Eigenvalue Decomposition...")
    delta, U = eigen(SB, SW)
    @assert all(isreal.(delta)) #&& all(delta .>= 0)
    info("Eigenvalue Decomposition...    Done!!");print("\r")
    delta, U = reverse(delta), reverse(U,dims=2)

    if numofcomponents == 0
        numofcomponents = sum(delta .> 1)
        println("\r\e[K\rNumber of components: $numofcomponents \e[K")
    elseif numofcomponents == -1
        plot(log.(delta[1:32]),linestyle=:solid, markershape=:circle, size=(1200,800))
        gui()
        print("\r\e[K\rEnter number of components: ")
        numofcomponents = parse(Int64, readline())
    end
    numofcomponents = min(numofcomponents, length(grouplist)-1)

    delta, U = delta[1:numofcomponents], U[:,1:numofcomponents]
    return Matrix{Float64}(U)

end

"""
    LDA(data::AbstractMatrix, labels::AbstractVector; outdim=0)

Given data and labels, return the LDA projection matrix.
"""
function LDA(data::AbstractMatrix, labels::AbstractVector; outdim=0)
    return LDA_projection_matrix(data, values(groupbyval(labels)), outdim)
end

"""
    buildTemplate(IVs, traces; nicv_th=nothing, POIe_left=0, POIe_right=0,
                               numofcomponents=0, priors=:uniform, printPOIlen=false)

Given the intermediate values (IVs) and traces, return the template.
"""
function buildTemplate(IVs::AbstractVector, traces::AbstractMatrix; nicv_th=nothing,
                       POIe_left=0, POIe_right=0, numofcomponents=0, priors=:uniform, printPOIlen=false)
    traceavg  = vec(mean(traces,dims=2))
    tracevar  = vec( var(traces,dims=2))
    trlen     = length(traceavg)
    groupdict = groupbyval(IVs)

    # find points of interests and extension (POIs/POIe)
    pois  = findPOI(IVs, traces; nicv_th, traceavg)
    pois  = expandPOI(pois, trlen, POIe_left, POIe_right)
    M_poi = sparse(Matrix{Float64}(I,trlen,trlen)[:,pois])
    print("\rCompress traces by NICV... \e[K")
    #memmap = Sys.free_memory() < sizeof(Traces)  # if system is HDD
    # use memory map if sizeof(traces_poi) exceeds 500MB
    memmap = (sizeof(traces)÷size(traces,1)*length(pois)) > 2^29 # 2^29 ≈ 500MB
    traces_poi = compresstraces(traces, pois; memmap)
    printPOIlen && println("\r\e[1A\r\e[36C -> POI trace length: $(length(pois))    \e[K")

    # dimension reduction using linear discriminant analysis (LDA)
    # then, project traces into LDA subspace
    print("\rLinear Discriminant Analysis... \e[K")
    U = LDA_projection_matrix(traces_poi, values(groupdict), numofcomponents)
    print("\rProjecting Traces onto LDA subspace... \e[K")
    traces_lda  = Matrix{Float64}(undef, size(U,2), size(traces_poi,2))
    LinearAlgebra.mul!(traces_lda, transpose(U), traces_poi); traces_poi = nothing; GC.gc()
    tr_lda_mean = vec(mean(traces_lda,dims=2))
    tr_lda_covM = cov(traces_lda, dims=2)

    # build MVG models for each indermedate value
    print("\rBuilding Multivariate Gaussian Models... \e[K")
    mvgs = Dict(Int16(l)=>MvNormal(vec(mean(traces_lda[:,gl],dims=2)),
                                        cov(traces_lda[:,gl],dims=2)) for (l,gl) in groupdict)
    labels = sort(collect(keys(groupdict)))
    if priors == :uniform
        priors = Dict(Int16(l)=>1.0/length(labels) for l in labels )
    elseif priors == :binomial
        d = Binomial(length(labels)-1,0.5)
        priors = Dict(Int16(l)=>p for (l,p) in zip(labels,pdf(d,support(d))))
    elseif priors isa Vector
        length(labels) == lenegth(priors) || error("length of `priors::Vector` doesn't match `labels`")
        priors = Dict(Int16(l)=>p for (l,p) in zip(labels,priors))
    elseif priors isa Dict
        sort(collect(keys(priors))) == labels || error("keys of `priors::Dict` doesn't match `labels`")
        priors = priors
    else # unknown prior, compute based on IV's distribution
        priors = Dict(Int16(l)=>length(groupdict[labels])/length(IVs) for l in labels )
    end

    # compute pooled covariance matrix (from data)
    print("\rCalculating pooled covariance Matrix... \e[K\r")
    for (l,gl) in groupdict
        view(traces_lda,:,gl) .-= mvgs[l].μ
    end
    pooled_cov_inv = inv(cov(traces_lda,dims=2))

    return Template(traceavg, tracevar, M_poi*U, tr_lda_mean, tr_lda_covM,
                    mvgs, priors, pooled_cov_inv)
end


### Template validation

## Percentile Rank (PR): the percentage of scores in its frequency distribution that are less than that score.
## check out: https://en.wikipedia.org/wiki/Percentile_rank
function ranknpercentile(t::Template, trace::AbstractVector, iv_true)
    labels = sort(collect(keys(t.mvgs)))
    lhs = likelihoods(t,trace; normalized=true)
    idx = sortperm(lhs,rev=true)
    pr, labels, lhs = 0.0, labels[idx], lhs[idx] # pr: percentile rank
    for (rank,l) in enumerate(labels)
        if l == iv_true
            pr  += lhs[rank]/2
            return rank, pr
        end
        pr  += lhs[rank]
    end
end

"""
    validate(t::Template, traces, IVs; show::Bool=false)

Validate the template with given traces and known intermediate values.
Return 1st order success rate, (mean, median, max) of Ranking and Percentile Rank.
"""
function validate(t::Template, traces, IVs; show::Bool=false)
    Ranks, PRs = zeros(Int,0), zeros(0)
    for (iv,tr) in zip(IVs,eachcol(traces))
        rank, pr = ranknpercentile(t,tr,iv)
        push!(Ranks,rank)
        push!(  PRs,  pr)
    end
    sr_1storder = mean(Ranks .== 1)     # first order success rate
    rank_stats  = mean(Ranks), median(Ranks), findmax(Ranks)[1] # (mean, median max)
    pr_stats    = mean(PRs),   median(PRs),   findmax(PRs)[1]

    if show
        @printf("Success Rate: %.2f%% |", sr_1storder*100)
        @printf("Ranking(median, max): (%.1f, %2i) |", rank_stats[2:3]...)
        @printf("PR(median, max): (%.4f, %.4f)    \n",   pr_stats[2:3]...)
    end
    return sr_1storder, rank_stats, pr_stats
end


### build Templates

"""
    runprofiling(IVs::AbstractMatrix, traces; nicv_th=nothing, POIe_left=0, POIe_right=0,
                 numofcomponents=0, priors=:uniform, outfile=nothing, nvalid=nothing)

Given the intermediate values (IVs) matrix and traces, run profiling and validation, and return a Vector of templates.
`priors` = :uniform, :binomial, ::Vector or ::Dict.
`nvalid` set the number of traces for validation, default is 2% of the total traces.
"""
function runprofiling(IVs::AbstractMatrix, traces; nicv_th=nothing, POIe_left=0, POIe_right=0,
                      numofcomponents=0, priors=:uniform, outfile=nothing, nvalid=nothing)
    # checking inputs
    IVs, traces = sizecheck(IVs, traces)
    outfile     = isnothing(outfile) ? joinpath(OUTDIR, outfile) : outfile
    isdir(dirname(outfile)) || mkpath(dirname(outfile))

    # split data into profiling set and validation set
    ntr    = size(traces,2)
    nvalid = isnothing(nvalid) ? max(1000,Int(ceil(size(traces,2)*0.02))) : Int(nvalid)
    range_p, range_v = 1:ntr-nvalid, ntr-nvalid+1:ntr
    tr_profile, tr_validate = view(traces,:,range_p), view(traces,:,range_v)
    iv_profile, iv_validate = view(   IVs,:,range_p), view(   IVs,:,range_v)

    # profiling
    templates = Vector{Template}(undef,0)
    for (byte,ivs) in enumerate(eachrow(iv_profile))
        println("Building Templates for byte: $byte/$(size(IVs,1)) \e[K")
        t = buildTemplate(ivs, tr_profile; nicv_th, POIe_left, POIe_right,
                                           numofcomponents, priors, printPOIlen=true)
        push!(templates, t)
        !isnothing(outfile) && writetemplate(outfile, t; byte, overwrite=(byte==1))
    end

    # validation
    for (byte,ivs) in enumerate(eachrow(iv_validate))
        @printf("byte: %2i -> ", byte)
        t = templates[byte]
        validate(t, tr_validate, ivs; show=true)
    end

    return templates
end


