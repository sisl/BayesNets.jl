#
# Likelihood Weighted Inference
#

"""
    weighted_built_in(inf, nsamples=500)

Likelihood weighted sampling using weighted sampling
"""
function weighted_built_in(inf::AbstractInferenceState, nsamples::Int=500)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence

    samples = rand(bn, WeightedSampler(evidence), nsamples)
    return by(samples, query, df -> DataFrame(potential = sum(df[:p])))
end

"""
    likelihood_weighting(inf, nsamples=500)

Approximates p(query|evidence) with `nsamples` likelihood weighted samples.

Since this uses a Factor, it is only efficient if the number of samples
is (signifcantly) greater than the number of possible instantiations for the
query variables
"""
function likelihood_weighting(inf::AbstractInferenceState, nsamples::Int=500)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence

    factor = Factor(query, map(n -> ncategories(bn, n), query))

    # if nodes are evidence
    evidence_mask = reduce(|, map(s -> nodes .== s, keys(evidence)))

    sample = Assignment()
    # add the evidence to the sample
    merge!(sample, evidence)

    # manual index into factor.potential since categorical implies Base.OneTo
    q_ind = similar(query, Int)

    for i = 1:nsamples
        w = 1.0

        for (cpd, is_ev) in zip(bn.cpds, evidence_mask)
            nn = name(cpd)
            d = cpd(sample)

            if !is_ev
                sample[nn] = rand(d)
            else
                w *= pdf(d, sample[nn])
            end
        end

        # pick out the query variables to index by
        for (i, q) in enumerate(query)
            q_ind[i] = sample[q]
        end

        factor.potential[q_ind...] += w
    end

    normalize!(factor)
    return factor
end

