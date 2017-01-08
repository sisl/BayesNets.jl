"""
Approximates p(query|evidence) with N weighted samples using likelihood
weighted sampling
"""
@with_kw type LikelihoodWeightingInference <: InferenceMethod
    nsamples::Int = 500
    grow::Bool = false
end
function infer(im::LikelihoodWeightingInference, bn::DiscreteBayesNet, query::Vector{NodeName}; evidence::Assignment=Assignment())
    if im.grow
        _infer_likelihood_weighting_grow(im, bn, query, evidence)
    else
        _infer_likelihood_weighting(im, bn, query, evidence)
    end
end

function _infer_likelihood_weighting(im::LikelihoodWeightingInference, bn::DiscreteBayesNet, query::Vector{NodeName}, evidence::Assignment)

    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:p]), im.nsamples)
    samples[:p] = 1
    sample = Assignment()

    for i in 1 : im.nsamples
        # will be in topological order because of
        #  _enforce_topological_order
        for cpd in bn.cpds
            nn = name(cpd)
            if haskey(evidence, nn)
                sample[nn] = evidence[nn]
                # update the weight with the pdf of the conditional
                # prob dist of a node given the currently sampled
                # values and the observed value for that node
                samples[i, :p] *= pdf(cpd, sample)
            else
                sample[nn] = rand(cpd, sample)
            end
        end

        # for some reason, you cannot set an entire row in a dataframe
        for q in query
            samples[i, q] = sample[q]
        end
    end

    samples = by(samples, query,
        df -> DataFrame(p = sum(df[:p])))
    samples[:p] /= sum(samples[:p])
    return samples
end

"""
If `samples` has `a`, replaces its value with f(samples[a, :p], v).
Else adds `a` and `v` to `samples`

`samples` must have a column called probabilities

All columns of `samples` must be in `a`, but not all columns of `a` must be
in `samples`

`f` should be able to take a DataFrames.DataArray as its first element
"""
function _update_samples!(samples::DataFrame, a::Assignment, v=1, f::Function=+)
    # copied this from filter in factors.jl
    # assume a has a variable for all columns except :probability
    mask = trues(nrow(samples))
    col_names = setdiff(names(samples), [:probability])

    for s in col_names
        mask &= (samples[s] .== a[s])
    end

    # hopefully there is only 1, but this still works else
    if any(mask)
        samples[mask, :probability] = f(samples[mask, :probability], v)
    else
        # get the assignment in the correct order for the dataframe
        new_row = [a[s] for s in col_names]
        push!(samples, @data(vcat(new_row, v)))
    end
end

"""
Likelihood weighting where the samples are stored in a DataFrame that grows
in size as more unique samples are observed.
"""
function _infer_likelihood_weighting_grow(im::LikelihoodWeightingInference, bn::BayesNet, query::Vector{Symbol}, evidence::Assignment)

    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:p]), 0)
    sample = Assignment()

    for i in 1 : im.nsamples
        wt = 1
        # will be in topological order because of
        #  _enforce_topological_order
        for cpd in bn.cpds
            nn = name(cpd)
            if haskey(evidence, nn)
                sample[nn] = evidence[nn]
                # update the weight with the pdf of the conditional
                # prob dist of a node given the currently sampled
                # values and the observed value for that node
                wt *= pdf(cpd, sample)
            else
                sample[nn] = rand(cpd, sample)
            end
        end

        # marginalize on the go
        # samples is over the query variables, sample is not, but it works
        _update_samples!(samples, sample, wt)
    end

    samples[:p] /= sum(samples[:p])
    return samples
end

