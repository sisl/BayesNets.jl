"""
Likelihood weighted sampling using weighted sampling
"""
function weighted_built_in(bn::BayesNet, query::Union{Vector{NodeName}, NodeName};
    evidence::Assignment=Assignment(),
    nsamples::Int=100,
    )

    samples = rand(bn, WeightedSampler(evidence), nsamples)
    return by(samples, query, df -> DataFrame(probability = sum(df[:p])))
end

"""
Approximates p(query|evidence) with N weighted samples using likelihood
weighted sampling
"""
function likelihood_weighting(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=500)
    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:probability]), N)
    samples[:probability] = 1
    sample = Assignment()

    for i = 1:N
        # will be in topological order because of
        #  _enforce_topological_order
        for cpd in bn.cpds
            nn = name(cpd)
            if haskey(evidence, nn)
                sample[nn] = evidence[nn]
                # update the weight with the pdf of the conditional
                # prob dist of a node given the currently sampled
                # values and the observed value for that node
                samples[i, :probability] *= pdf(cpd, sample)
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
        df -> DataFrame(probability = sum(df[:probability])))
    samples[:probability] /= sum(samples[:probability])
    return samples
end

"""
If `samples` has `a`, replaces its value with f(samples[a, :probability], v).
Else adds `a` and `v` to `samples`

`samples` must have a column called probabiliteis

All columns of `samples` must be in `a`, but not all columns of `a` must be
in `samples`

`f` should be able to take a DataFrames.DataArray as its first element
"""
function update_samples(samples::DataFrame, a::Assignment, v=1, f::Function=+)
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
function likelihood_weighting_grow(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=500)
    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:probability]), 0)
    sample = Assignment()

    for i = 1:N
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
        update_samples(samples, sample, wt)
    end

    samples[:probability] /= sum(samples[:probability])
    return samples
end

# versions to accept just one query variable, instead of a vector
function likelihood_weighting(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=500)
    return likelihood_weighting(bn, [query]; evidence=evidence, N=N)
end

function likelihood_weighting_grow(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=500)
    return likelihood_weighting_grow(bn, [query]; evidence=evidence, N=N)
end

