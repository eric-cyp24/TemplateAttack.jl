
### template matching

"""
    match(t::Template, trace::AbstractVector)

Template matching, return a vector of normalized likelihoods, ordering by the sorted labels.
"""
function match(t::Template, trace::AbstractVector)
    labels,lhs = sort(collect(keys(t.mvgs))), likelihoods(t,trace)
    idx = sortperm(lhs,rev=true)
    return collect(zip(labels,lhs))[idx]
end

function match_pooled_cov(t::Template, trace::AbstractVector; uselogpdf::Bool=false)
    trace    = ndims(t)==size(traces,1) ? trace : LinearAlgebra.mul!(Vector{Float64}(undef, ndims(t)), transpose(t.ProjMatrix), trace)
    d        = trace .- stack(t.mus)
    logprobs = -0.5*(transpose(d)*t.pooled_cov_inv*d)
    if uselogpdf
        return logprobs
    else
        probs = exp(logpdf) .* t.probs
        return probs / sum(probs)
    end
end

"""
    best_match(t::Template, trace::AbstractVector)
    best_match(t::Template, traces::AbstractMatrix)

Find the best match label(s) given trace(s).
"""
function best_match(t::Template, trace::AbstractVector)
    return sort(collect(keys(t.priors)))[argmax(likelihoods(t,trace))]
end
function best_match(t::Template, traces::AbstractMatrix)
    return sort(collect(keys(t.priors)))[argmax.(eachcol(likelihoods(t,traces)))]
end



## Likelihoods
"""
    likelihoods(t::Template,  trace::AbstractVector; normalized::Bool=true)
    likelihoods(t::Template, traces::AbstractMatrix; normalized::Bool=true)

Given template and trace(s), return the Vector/Matrix of likelihoods
with the order of sorted template labels.
"""
function likelihoods(t::Template, trace::AbstractVector; normalized::Bool=true)
    trace = ndims(t)==size(trace,1) ? trace : LinearAlgebra.mul!(Vector{Float64}(undef, ndims(t)), transpose(t.ProjMatrix), trace)
    lhs = [t.priors[k]*pdf(t.mvgs[k],trace) for k in sort(collect(keys(t.mvgs)))]
    return normalized ? lhs / sum(lhs) : lhs
end
function likelihoods(t::Template, traces::AbstractMatrix; normalized::Bool=true)
    labels = sort(collect(keys(t.mvgs)))
    traces = ndims(t)==size(traces,1) ? traces :
             LinearAlgebra.mul!(Matrix{Float64}(undef, ndims(t), size(traces,2)), transpose(t.ProjMatrix), traces)
    lhs = Matrix{Float64}(undef,length(labels),size(traces)[2])
    for (i,l) in enumerate(labels)
        lhs[i,:] = t.priors[l]*pdf(t.mvgs[l],traces)
    end
    return normalized ? lhs ./ sum(lhs,dims=1) : lhs
end

function loglikelihoods(t::Template, trace::AbstractVector)
    trace = ndims(t)==size(traces,1) ? trace : LinearAlgebra.mul!(Vector{Float64}(undef, ndims(t)), transpose(t.ProjMatrix), trace)
    lhs = [log(t.priors[k])*logpdf(t.mvgs[k],trace) for k in sort(collect(keys(t.mvgs)))]
    return lhs
end
function loglikelihoods(t::Template, traces::AbstractMatrix)
    labels = sort(collect(keys(t.mvgs)))
    traces = ndims(t)==size(traces,1) ? traces :
             LinearAlgebra.mul!(Matrix{Float64}(undef, ndims(t), size(traces,2)), transpose(t.ProjMatrix), traces)
    lhs = Matrix{Float64}(undef,length(labels),size(traces)[2])
    for (i,l) in enumerate(labels)
        lhs[i,:] = log(t.priors[l])*logpdf(t.mvgs[l],traces)
    end
    return lhs
end


## Key Guessing Vector
"""
    key_guessing(t::Template,  trace::AbstractVector)
    key_guessing(t::Template, traces::AbstractMatrix)

Returns a vector of key candidates ranked by decreasing order of likelihoods from templates.
"""
function key_guessing(t::Template, trace::AbstractVector)
    return sort(collect(keys(t.mvgs)))[sortperm(likelihoods(t,trace;normalized=false),rev=true)]
end
function key_guessing(t::Template, traces::AbstractMatrix)
    labels = sort(collect(keys(t.mvgs)))
    lhs    = likelihoods(t,traces;normalized=false)
    key_guesses = similar(lhs, eltype(labels))
    for (i,lh) in enumerate(eachcol(lhs))
        view(key_guesses,:,i) .= @view labels[sortperm(lh,rev=true)]
    end
    return key_guesses
end


## Success Rate
"""
    success_rate(key_guesses::AbstractMatrix, true_keys::AbstractVector; order::Integer=1)
    success_rate(t::Template, traces::AbstractMatrix, true_keys::AbstractVector; order::Integer=1)

Success rate of order o is the average probability that the secret key is located within the first o elements of the key guessing vector. The default `order=1` is the 1st-order success rate.
"""
function success_rate(key_guesses::AbstractMatrix, true_keys::AbstractVector; order::Integer=1)
    # size check
    if size(key_guesses,2) != length(true_keys)
        msg = "key_guesses length $(size(key_guesses)) doesn't match true_keys length $(length(true_keys))"
        throw(DimensionMismatch(msg))
    end
    return mean((x->x[1] in x[2]).(zip(true_keys,eachcol(view(key_guesses,1:order,:)))))
end
function success_rate(t::Template, traces::AbstractMatrix, true_keys::AbstractVector; order::Integer=1)
    return success_rate(key_guessing(t,traces),true_keys; order)
end


## Guessing Entropy
"""
    guessing_entropy(key_guesses::AbstractMatrix, true_keys::AbstractVector)
    guessing_entropy(t::Template, traces::AbstractMatrix, true_keys::AbstractVector)

Guessing Entropy represents the average position of the `true_key` k*.
"""
function guessing_entropy(key_guesses::AbstractMatrix, true_keys::AbstractVector)
    # size check
    if size(key_guesses,2) != length(true_keys)
        msg = "key_guesses length $(size(key_guesses)) doesn't match true_keys length $(length(true_keys))"
        throw(DimensionMismatch(msg))
    end
    return mean(x->findfirst(isequal(x[1]),x[2]),zip(true_keys,eachcol(key_guesses)))
end
function guessing_entropy(t::Template, traces::AbstractMatrix, true_keys::AbstractVector)
    return guessing_entropy(key_guessing(t,traces), true_keys)
end




