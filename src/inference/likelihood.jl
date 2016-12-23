#
# Likelihood Weighted Inference
#
# Likelihood weighting and associated functions (including custom rand)

"""
Custom rand! returns a sample and a weight. No DataFrame.
Faster? Maybe.
Convoluted? Maybe.
`non_evidence_cpds` must be in topological order (e.g. from bn.names)
Sample must have evidence instances in it already.
"""
@inline function _rand!(sample::Assignment,
        non_evidence_cpds=Vector{CPD},
        evidence_cpds::Vector{CPD})
    w = 1.0

    for cpd in non_evidence_cpds
        nn = name(cpd)
        sample[nn] = rand(cpd, sample)
    end

    for cpd in evidence_cpds
        w *= pdf(cpd, sample)
    end

    return (sample, w)
end

"""
    likelihood_weighting(inf, nsamples=500)

Approximates p(query|evidence) with `nsamples` likelihood weighted samples.
Since this uses a Factor, it is only efficient if the number of samples
is (signifcantly) greater than the number of possible instantiations for the
query variables
"""
function likelihood_weighting_inf(inf::AbstractInferenceState, nsamples::Int=500)
    nodes = names(bn)
    query = names(inf.factor)
    evidence = inf.evidence
    wt_sampler = WeightedSampler(evidence)

    # which nodes are evidence, used for cpds, since cpds and names
    # have the same order
    ev_mask = reduce(|, map(s -> nodes .== s, keys(evidence)))
    evidence_cpds = bn.cpds[ev_mask]
    non_evidence_cpds = bn.cpds[!ev_mask]

    sample = Assignment()
    # add the evidence to the sample
    merge!(sample, evidence)

    # manual index into factor.f since categorical = Base.OneTo
    q_ind = similar(query, Int)

    for i = 1:nsamples
        (sample, w) = _rand(sample, non_evidence_cpds, evidence_cpds)

        for (i, q) in enumerate(query)
            q_ind[i] = sample[q]
        end

        inf.factor.v[q_ind...] += w
    end

    return inf
end

function likelihood_weighting(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), nsamples::Int=500)
    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:probability]), nsamples)
    samples[:probability] = 1
    sample = Assignment()

    for i = 1:nsamples
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

# versions to accept just one query variable, instead of a vector
function likelihood_weighting(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), nsamples::Int=500)
    return likelihood_weighting(bn, [query]; evidence=evidence, nsamples=nsamples)
end

