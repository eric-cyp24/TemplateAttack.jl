

mutable struct Template
    # Raw Trace space
    TraceMean::Vector{Float64}
    TraceVar::Vector{Float64}
    ProjMatrix::Matrix{Float64} # Raw-to-LDA_Subspace
    # LDA Subspace
    mean::Vector{Float64}       # overall mean of the profiling traces in LDA-space
    covMatrix::Matrix{Float64}  # overall covM of the profiling traces in LDA-space
    # MVG models
    labels::Vector{Integer}
    mus::Vector{Vector{Float64}}
    sigmas::Vector{Matrix{Float64}}
    mvgs::Vector{MvNormal}
    probs::Vector{Float64}
    pooled_cov_inv::Matrix{Float64} # for Pooled Covariance Matrix

    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, labels, mus::Matrix{T}, 
                      sigmas::Array{T,3}, probs, pooled_cov_inv=nothing) where T<:AbstractFloat
        ProjMatrix = size(ProjMatrix)[1] == length(TraceMean) ? ProjMatrix : Array(ProjMatrix')
        idx    = sortperm(labels)
        labels = labels[idx]
        mus    = eachcol(mus)[idx]
        sigmas = eachslice(sigmas, dims=3)[idx]
        mvgs   = [MvNormal(m,s) for (m,s) in zip(mus, sigmas)]
        probs  = sum(probs) == 1 ? probs : (probs ./ sum(probs))
        if isnothing(pooled_cov_inv)
            pooled_cov_inv = inv(sum(probs .* eachslice(sigmas, dims=3)))
        end
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix,
            labels, mus, sigmas, mvgs, probs, pooled_cov_inv)
    end
end

### template add data
function addtemplate(t::Template, name, mu, sigma)
    push!(labels, name)
    push!(mus, mu)
    push!(sigmas, sigma)
    push!(mvgs, MvNormal(mu, sigma))
end

function addpooledcovMatrix(t::Template, cov_pooled)
    t.pooled_cov_inv = inv(cov_pooled)
end



### template load/dump

#templatefields = String.(fieldnames(Template))
templatefields = ("TraceMean", "TraceVar", "ProjMatrix", "mean", "covMatrix",
                  "labels", "mus", "sigmas", "probs", "pooled_cov_inv")

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

### template matching

"""
    match(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    match(t::Template, traces::AbstractMatrix; uselogpdf::Bool=false)

template matching
"""
function match(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    trace   = t.ProjMatrix' * trace
    pdfeval = uselogpdf ? logpdf : pdf
    probs   = [p*pdfeval(mvg,trace) for (p,mvg) in zip(t.probs, t.mvgs)]
    return probs / sum(probs)
end

function match(t::Template, traces::AbstractMatrix; uselogpdf::Bool=false)
    return [match(t,tr) for tr in eachcol(traces)]
end

function match_pooled_cov(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    trace    = t.ProjMatrix' * trace
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

function best_match(t::Template, traces::AbstractMatrix)
    return [best_match(t,tr) for tr in eachcol(traces)]
end




