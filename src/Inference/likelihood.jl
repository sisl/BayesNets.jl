"""
Approximates p(query|evidence) with N weighted samples using likelihood
weighted sampling
"""
@with_kw struct LikelihoodWeightingInference <: InferenceMethod
    nsamples::Int = 500
end

"""
Approximates p(query|evidence) with `nsamples` likelihood weighted samples.

Since this uses a Factor, it is only efficient if the number of samples
is (signifcantly) greater than the number of possible instantiations for the
query variables
"""
function infer(im::LikelihoodWeightingInference, inf::InferenceState{BN}) where {BN<:DiscreteBayesNet}
    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence

    ϕ = Factor(query, map(n -> ncategories(bn, n), query))

    # if nodes are evidence
    evidence_mask = [haskey(evidence, s) for s in nodes]

    sample = Assignment()
    # add the evidence to the sample
    merge!(sample, evidence)

    # manual index into ϕ.potential since categorical implies Base.OneTo
    q_ind = similar(query, Int)

    for i in 1:im.nsamples
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

        ϕ.potential[q_ind...] += w
    end

    normalize!(ϕ)
    return ϕ
end

