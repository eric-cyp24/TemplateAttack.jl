using StatsPlots, HDF5, Statistics, Distributions #, LinearAlgebra
using Distributions: MvNormal
using Npy, EMAlgorithm

mutable struct Template
    # Raw Trace space
    TraceMean::Vector{Float64}
    TraceVar::Vector{Float64}
    ProjMatrix::Matrix{Float64} # Raw-to-LDA_Subspace
    # LDA Subspace
    mean::Vector{Float64}
    covMatrix::Matrix{Float64}
    # MVG models
    labels::Vector{Integer}
    mus::Vector{Vector{Float64}}
    sigmas::Vector{Matrix{Float64}}
    mvgs::Vector{MvNormal}
    probs::Vector{Float64}
    pooled_cov_inv::Matrix{Float64} # for Pooled Covariance Matrix

    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, labels,
                      mus::Matrix{T}, sigmas::Array{T,3}, probs, pooled_cov_inv) where T <: AbstractFloat
        idx    = sortperm(labels)
        labels = labels[idx]
        mus    = eachcol(mus)[idx]
        sigmas = eachslice(sigmas, dims=3)[idx]
        mvgs   = [MvNormal(m,s) for (m,s) in zip(mus, sigmas)]
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix,
            labels, mus, sigmas, mvgs, probs, pooled_cov_inv)
    end
end

#templatefields = String.(fieldnames(Template))
templatefields = ("TraceMean", "TraceVar", "ProjMatrix", "mean", "covMatrix",
                  "labels", "mus", "sigmas", "probs", "pooled_cov_inv")

function match(t::Template, traces::AbstractMatrix; uselogpdf::Bool=false)
    return [match(t,tr) for tr in eachcol(traces)]
end

function match(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    trace   = t.ProjMatrix * trace
    pdfeval = uselogpdf ? logpdf : pdf
    probs   = [p*pdfeval(mvg,trace) for (p,mvg) in zip(t.probs, t.mvgs)]
    return probs / sum(probs)
end

function match_pooled_cov(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    trace    = t.ProjMatrix * trace
    d        = trace .- stack(t.mus)
    logprobs = -0.5*(transpose(d)*t.pooled_cov_inv*d)
    if uselogpdf
        return logprobs
    else
        probs = exp(logpdf) .* t.probs
        return probs / sum(probs)
    end
end


function best_match(t::Template, trace::AbstractVector)
    return t.labels[argmax(match(t,trace))]
end
best_match(t::Template, traces::AbstractMatrix) = [best_match(t,tr) for tr in eachcol(traces)]


load_dataset(h5group, h5dataset) = h5dataset in keys(h5group) ? read_dataset(h5group, h5dataset) : nothing
function loadtemplate(filename::AbstractString; byte=0)
    t = h5open(filename, "r") do h5
        g = open_group(h5, "Templates/byte $byte")
        return [load_dataset(g, String(n)) for n in templatefields]
    end
    return Template(t...)
end

function writetemplate(filename::AbstractString, template::Template; group_path="Templates/byte 0")
    h5open(filename, "cw") do h5
        try delete_object(h5, group_path) catch e end
        g = create_group(h5, group_path)
        for n in fieldnames(Template)
            if n == :mus || n == :sigmas
                write_dataset(g,String(n),stack(getproperty(template,n)))
            elseif n == :mvgs
                continue
                mvgs = getproperty(template,n)
                write_dataset(g,    "mus", stack([g.μ for g in mvgs]))
                write_dataset(g, "sigmas", stack([g.Σ for g in mvgs]))
            elseif n == :labels
                write_dataset(g, String(n), convert.(Int32, getproperty(template,n)))
            else
                write_dataset(g, String(n), getproperty(template,n))
            end
        end
    end
end

function Template2GMM(t::Template)
    return GaussianMixtureModel(t.mus, t.sigmas, t.probs)
end

EMAlgorithm.GaussianMixtureModel(t::Template) = GaussianMixtureModel(t.mus, t.sigmas, t.probs)

function adjtemplate!(t::Template, traces::AbstractMatrix, num_epoch::Integer=50; δ=10e-6)
    if size(traces,1) == size(t.ProjMatrix,2)
        traces = t.ProjMatrix * traces
    end
    # templates->gmm && re-center -> EM Alg
    gmm = Template2GMM(t) + vec(mean(traces,dims=2) - t.mean)
    llh = emalgorithm_fixedweight_mprocess!(gmm, traces, num_epoch; δ)
    # update MVG templates' params
    for i in 1:length(gmm)
        t.mvgs[i]   = gmm.components[i]
        t.mus[i]    = gmm.components[i].μ
        t.sigmas[i] = gmm.components[i].Σ
    end
    return gmm
end

function loadtrace(filename::AbstractString)
    if isnpy(filename)
        return loadnpy(filename;numpy_order=false)
    elseif HDF5.ifhdf5(filename)
        h5open(filename, "r") do
            return read(f["Data/Traces"])
        end
    end
end


function testadjtemplate(tracefile, templatefile; axes=[1,2])
    Traces   = loadtrace(tracefile)
    template = loadtemplate(templatefile, byte=0)

    plot(;size=(1200,900))
    plotEM!(Traces, Template2GMM(template); axes, show=false, linestyle=:dash)
    sleep(3)

    adjtemplate!(template, Traces)
    writetemplate(filename, template; group_path="Template/byte 0")

    # plot result
    plot(;size=(1200,900))
    glist = [[] for _ in 1:length(Set(Ans))]
    n_min = min(Set(Ans)...)
    for (i,n) in enumerate(Ans)
        push!(glist[n+1-n_min], i)
    end
    for (n,gl) in enumerate(glist)
        plotdatascatter!(Traces[:,gl]; axes, color=n, show=false)
    end
    plotGMM!(gmm; axes)
end


## test ###
# Input : template, traces[, Ans]
# Output: success rate / Probs
tracefiles    = ["test/AdjTA/Board_$(brd)_test/traces_test_lanczos2_25_proc.npy" for brd in "ABCO"]
ivfiles       = ["test/AdjTA/Board_$(brd)_test/A_test_proc.npy" for brd in "ABCO"]
templatefiles = ["test/Board_$(brd)0_A.h5" for brd in "ABCO"]

function test_sr(tracefile, ivfile, templatefile; adj::Bool=false, num_epoch::Integer=50)
    print("\rLoading Traces & IVs...   ")
    Traces = loadtrace(tracefile)
    IVs    = loadnpy(ivfile;numpy_order=false)
    print("\rLoading Templates...      ")
    numtpl = length(open_group(h5open(templatefile),"Templates"))
    Templates = [loadtemplate(templatefile,byte=i) for i in 0:numtpl-1]

    if ndims(Traces) == 3
        a,b,c = size(Traces)
        Traces = reshape(Traces, a, b*c)
    end

    # EMalg for adj template
    if adj
        for (i,t) in enumerate(Templates)
            print("\rAdjusting Templates byte $i with EM Algorithm...    \r")
            adjtemplate!(t,Traces)
        end
    end

    print("\rTemplate matching...                             ")
    Probs_matrix = stack([[match(t,tr) for t in Templates] for tr in eachcol(Traces)])
    if ndims(loadtrace(tracefile))==3
        Probs_matrix = reshape(Probs_matrix, length(Templates)*b,c)
    end
    print("\rComputing success rate...      ")
    IV_guess = Templates[1].labels[argmax.(Probs_matrix)]
    correct_guess = (IV_guess .== IVs)
    Success_rate = sum(correct_guess)/length(correct_guess)
    println("Success Rate: $Success_rate       ")
end









