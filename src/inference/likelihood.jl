#
# Likelihood Weighted Inference
#
# Likelihood weighting and associated functions (including custom rand)

"""
Likelihood weighted sampling using weighted sampling
"""
function weighted_built_in(inf::AbstractInferenceState, nsamples::Int=500)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence

    samples = rand(bn, WeightedSampler(evidence), nsamples)
    return by(samples, query, df -> DataFrame(probability = sum(df[:p])))
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

    factor = Factors.Factor(query,
                map(n -> ncategories(bn, n), query), Float64)

    # if nodes are evidence
    #  for cpd selection; cpds and names have the same order
    ev_mask = reduce(|, map(s -> nodes .== s, keys(evidence)))
    evidence_cpds = bn.cpds[ev_mask]
    non_evidence_cpds = bn.cpds[!ev_mask]

    sample = Assignment()
    # add the evidence to the sample
    merge!(sample, evidence)

    # manual index into factor.f since categorical implies Base.OneTo
    q_ind = similar(query, Int)

    for i = 1:nsamples
        w = 1.0

        for cpd in non_evidence_cpds
            nn = name(cpd)
            sample[nn] = rand(_get_pdf(cpd, sample))
        end

        for cpd in evidence_cpds
            w *= pdf(_get_pdf(cpd), sample[cpd.name])
        end

        # pick out the query variables to index by
        for (i, q) in enumerate(query)
            q_ind[i] = sample[q]
        end

        factor.v[q_ind...] += w
    end

    normalize!(factor)
    return factor
end

@inline function _get_pdf(cpd::CategoricalCPD{Categorical{Float64}}, 
        sample::Dict{Symbol, Any})
    ind = [p -> sample[p] for p in cpd.parents]
    i = sub2ind((cpd.parental_ncategories...), ind...)

    return cpd.distributions[i]
end


