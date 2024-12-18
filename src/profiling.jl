

### Template profiling

"""
    findPOI(IVs, traces; nicv_th=nothing, traceavg=nothing)

use NICV to find POIs, by setting a NICV threshold.
"""
function findPOI(IVs::AbstractVector, traces; nicv_th=nothing, traceavg=nothing)
    traceavg = isnothing(traceavg) ? view(traces,:,1:50) : traceavg
    print("Calculating NICV...                       \r")
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
function expandPOI(pois::AbstractVector, trlen, POIe_left, POIe_right)
    pois_set = Set(pois)
    for i in pois
        for p in i-POIe_left:i+POIe_right
            0 < p ≤ trlen && push!(pois_set, p)
        end
    end
    return sort(collect(pois_set))
end

function compresstraces(traces, pois::AbstractVector, tempfile::AbstractString=TMPFILE)
    # use memory map if sizeof(traces_poi) exceeds 500MB
    if log2(sizeof(traces)*length(pois)/size(traces,1))>29
        print("writing to tmp...                \r")
        open(tempfile,"w+") do f
            traces_poi = mmap(f, Matrix{eltype(traces)}, (length(pois),size(traces,2)))
            traces_poi[:] = view(traces,pois,:)
        end
        traces_poi = open(tempfile,"r") do f
            mmap(f, Matrix{eltype(traces)}, (length(pois),size(traces,2)))
        end
        return traces_poi
    else
        return traces[pois,:]
    end
end

function LDA_projection_matrix(traces::AbstractMatrix, grouplist, numofcomponents=0)
    info = "Linear Discriminant Analysis...   "
    TracesMean = vec(mean(traces,dims=2))
    TraceMeans = stack([vec(mean(view(traces,:,gl), dims=2)) for gl in grouplist])

    print(info,"finding Sb...       \r")
    Nt = sqrt.([length(gl) for gl in grouplist])'
    T = (TraceMeans .- TracesMean) .* Nt
    SB = T * T'

    print(info,"finding Sw...           \r")
    T = similar(traces) # large memory allocation is slow...
    for (i,gl) in enumerate(grouplist)
        view(T,:,gl)[:] = view(traces,:,gl) .- TraceMeans[:,i]
    end
    #grouplist = collect(grouplist)
    #@sync Threads.@threads for i in 1:length(grouplist)  # TODO: multithreading
    #    view(T,:,grouplist[i])[:] = view(traces,:,grouplist[i]) .- TraceMeans[:,i]
    #end
    SW = T * T'
    T = nothing; GC.gc(); # release memory

    print(info,"Eigenvalue Decomposition...")
    delta, U = eigen(SB, SW)
    print("Done!!               \r")
    delta, U = reverse(delta), reverse(U,dims=2)

    if numofcomponents == 0
        numofcomponents = sum(delta .> 1)
        print("\r                                                                    \r")
        println("Number of components: $numofcomponents           ")
    elseif numofcomponents == -1
        plot(log.(delta[1:32]),linestyle=:solid, markershape=:circle, size=(1200,800))
        gui()
        print("\r                                                                    \r")
        print("Enter number of components: ")
        numofcomponents = parse(Int64, readline())
    end
    numofcomponents = min(numofcomponents, length(grouplist)-1)

    delta, U = delta[1:numofcomponents], U[:,1:numofcomponents]
    return U

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
                               numofcomponents=0, priors=:uniform, tempfile=nothing)

Given the intermediate values (IVs) and traces, return the template.
"""
function buildTemplate(IVs::AbstractVector, traces::AbstractMatrix; nicv_th=nothing, POIe_left=0, 
                       POIe_right=0, numofcomponents=0, priors=:uniform)
    traceavg  = vec(mean(traces,dims=2))
    tracevar  = vec( var(traces,dims=2))
    trlen     = length(traceavg)
    groupdict = groupbyval(IVs)
    
    # find points of interests and extension (POIs/POIe)
    pois  = findPOI(IVs, traces; nicv_th, traceavg)
    pois  = expandPOI(pois, trlen, POIe_left, POIe_right)
    M_poi = sparse(Matrix(I,trlen,trlen)[:,pois])
    print("Compress traces by NICV...         \r")
    traces_poi = compresstraces(traces, pois)
    println("\r\e[1A\r\e[32C -> POI trace length: $(length(pois))    ")

    # dimension reduction using linear discriminant analysis (LDA)
    # then, project traces into LDA subspace
    print("Linear Discriminant Analysis...                             \r")
    U = LDA_projection_matrix(traces_poi, values(groupdict), numofcomponents)
    print("Projecting Traces onto LDA subspace...")
    print("                                    \r")
    traces_lda  = U' * traces_poi
    tr_lda_mean = vec(mean(traces_lda,dims=2))
    tr_lda_covM = cov(traces_lda, dims=2)

    # build MVG models for each indermedate value
    print("Building Multivariate Gaussian Models...                    \r")
    mvgs = Dict(Int16(l)=>MvNormal(vec(mean(traces_lda[:,gl],dims=2)),
                                        cov(traces_lda[:,gl],dims=2)) for (l,gl) in groupdict)
    labels = sort(collect(keys(groupdict)))
    if priors == :uniform
        priors = Dict(Int16(l)=>1.0/length(labels) for l in labels )
    elseif priors == :binomial
        d = Binomial(length(labels)-1,0.5)
        priors = Dict(Int16(l)=>p for (l,p) in zip(labels,pdf(d,support(d))))
    elseif priors isa Vector
        priors = Dict(Int16(l)=>p for (l,p) in zip(labels,priors))
    elseif priors isa Dict
        priors = priors
    else
        priors = Dict(Int16(l)=>1.0/length(labels) for l in labels )
    end

    # compute pooled covariance matrix (from data)
    print("Calculating pooled covariance Matrix...                     \r")
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
                 numofcomponents=0, priors=:uniform, outfile=nothing)

Given the intermediate values (IVs) matrix and traces, run profiling and validation, and return a Vector of templates.
`priors` = :uniform, :binomial, ::Vector or ::Dict
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
        println("Building Templates for byte: $byte                                         ")
        t = buildTemplate(ivs, tr_profile; nicv_th, POIe_left, POIe_right, 
                                           numofcomponents, priors)
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


