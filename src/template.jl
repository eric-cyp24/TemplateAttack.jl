

mutable struct Template
    # Raw Trace space
    TraceMean::Vector{Float32}
    TraceVar::Vector{Float32}
    ProjMatrix::Matrix{Float64} # Raw-to-LDA_Subspace
    # LDA Subspace
    mean::Vector{Float64}       # overall mean of the profiling traces in LDA-space
    covMatrix::Matrix{Float64}  # overall covM of the profiling traces in LDA-space
    # MVG models
    mvgs::Dict{Int16, Distributions.FullNormal}
    priors::Dict{Int16, Float64}
    pooled_cov_inv::Matrix{Float64} # for Pooled Covariance Matrix

    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, mvgs, priors, pooled_cov_inv)
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, mvgs, priors, pooled_cov_inv)
    end
    
    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix)
        mvgs = Dict{Int16, Distributions.FullNormal}()
        priors = Dict{Int16, Float64}()
        pooled_cov_inv = Matrix{Float64}() 
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, mvgs, priors, pooled_cov_inv)
    end
    
    function Template(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, labels, mus::Matrix{T}, 
                      sigmas::Array{T,3}, priors, pooled_cov_inv=nothing) where T<:AbstractFloat
        ProjMatrix = size(ProjMatrix)[1] == length(TraceMean) ? ProjMatrix : Array(ProjMatrix')
        mvgs   = Dict(labels[i]=>MvNormal(mus[:,i],sigmas[:,:,i]) for i in 1:length(labels))
        priors  = sum(priors) == 1 ? priors : (priors ./ sum(priors))
        if isnothing(pooled_cov_inv)
            pooled_cov_inv = inv(sum(priors .* eachslice(sigmas, dims=3)))
        end
        priors  = Dict(labels[i]=>priors[i] for i in 1:length(labels))
        new(TraceMean, TraceVar, ProjMatrix, mean, covMatrix, mvgs, priors, pooled_cov_inv)
    end

end

## some Base functions

Base.copy(t::Template)   = Template([copy(getfield(t,n)) for n in fieldnames(Template)]...)
Base.length(t::Template) = length(t.mvgs)
Base.size(t::Template)   = (length(t.mvgs), size(t.ProjMatrix,2))
Base.ndims(t::Template)  = size(t.ProjMatrix,2)
function Base.:(==)(t1::Template,t2::Template)
    return all([getfield(t1,n)==getfield(t2,n) for n in fieldnames(Template)])
end


### template add data ... might remove this in the future...
function addtemplate!(t::Template, label, mu, sigma)
    t.mvgs[label] = MvNormal(mu,sigma)
end

function addprior!(t::Template, label, prior)
    t.priors[label] = prior
end

function addpooledcovMatrix!(t::Template, cov_pooled=nothing)
    if isnothing(cov_pooled)
        p_sum = sum(values(t.priors))
        for k in keys(t.priors) t.priors[k] /= p_sum end
        cov_polled = sum([t.priors[k]*t.mvgs[k].Σ for k in keys(mvgs)])
    else
        t.pooled_cov_inv = inv(cov_pooled)
    end
end

### template load/dump

templatefields = ("TraceMean", "TraceVar", "ProjMatrix", "mean", "covMatrix",
                  "labels", "mus", "sigmas", "priors", "pooled_cov_inv")

"""
    loadtemplate(filename::AbstractString; group_path="Templates/" byte=0)

Load template from .h5 file.
"""
function loadtemplate(filename::AbstractString; group_path="Templates/", byte=0)
    template_path = joinpath(group_path,"byte $byte")
    t = h5open(filename, "r") do h5
        g = open_group(h5, template_path)
        return [read_dataset(g, String(n)) for n in templatefields] # do block result return to t
    end
    return Template(t...)
end

"""
    writetemplate(filename::AbstractString, t::Template; group_path::AbstractString="Templates/", byte=0, overwrite::Bool=false)

Write template to .h5 file.
"""
function writetemplate(filename::AbstractString, t::Template; group_path::AbstractString="Templates/", byte=0, overwrite::Bool=false)
    template_path = joinpath(group_path,"byte $byte")
    h5open(filename, overwrite ? "w" : "cw") do h5
        try delete_object(h5, template_path) catch e end
        g = create_group(h5, template_path)
        idx    = sortperm(collect(keys(t.mvgs)))
        labels = collect(keys(t.mvgs))[idx]
        mus    = stack([t.mvgs[l].μ for l in labels])
        sigmas = stack([t.mvgs[l].Σ for l in labels])
        priors = [t.priors[l] for l in labels]
        for n in fieldnames(Template)
            if n == :mvgs
                write_dataset(g, "labels", labels)
                write_dataset(g,    "mus", mus)
                write_dataset(g, "sigmas", sigmas)
            elseif n == :priors
                write_dataset(g, String(n), priors)
            else
                write_dataset(g, String(n), getproperty(t,n))
            end
        end
    end
end




