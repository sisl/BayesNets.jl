"""
Abstract type for sampling with:
 * `Random.rand(BayesNet, BayesNetSampler)`
 * `Random.rand(BayesNet, BayesNetSampler, nsamples)`
 * `Random.rand!(Assignment, BayesNet, BayesNetSampler)`
"""
abstract type BayesNetSampler end

"""
Overwrites assignment with a sample from bn using the sampler
"""
Random.rand!(a::Assignment, bn::BayesNet, sampler::BayesNetSampler) =
        error("rand! not implemented for $(typeof(sampler))")

"""
Returns an assignment sampled from the bn using the provided sampler
"""
Random.rand(bn::BayesNet, sampler::BayesNetSampler) = rand!(Assignment(), bn, sampler)

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows.
"""
function Random.rand(bn::BayesNet, sampler::BayesNetSampler, nsamples::Integer)

    a = rand(bn, sampler)
    df = DataFrame()
    for cpd in bn.cpds
        df[!, name(cpd)] = Array{typeof(a[name(cpd)])}(undef, nsamples)
    end

    for i in 1:nsamples
        rand!(a, bn, sampler)
        for cpd in bn.cpds
            n = name(cpd)
            df[i, n] = a[n]
        end
    end

    df
end

#
# Direct Sampling
#
"""
Straightforward sampling from a BayesNet.
The default sampler.
"""
struct DirectSampler <: BayesNetSampler end

function Random.rand!(a::Assignment, bn::BayesNet, sampler::DirectSampler)
    for cpd in bn.cpds
        a[name(cpd)] = rand(cpd, a)
    end
    a
end

Random.rand!(a::Assignment, bn::BayesNet) = rand!(a, bn, DirectSampler())

Random.rand(bn::BayesNet, nsamples::Integer) = rand(bn, DirectSampler(), nsamples)

Random.rand(bn::BayesNet) = rand(bn, DirectSampler())

#
# Rejection Sampling
#
"""
Rejection Sampling in which the assignments are forced to be consistent with the provided values.
Each sampler is attempted at most `max_nsamples` times before returning an empty assignment.
"""
struct RejectionSampler <: BayesNetSampler
    evidence::Assignment
    max_nsamples::Int
end
RejectionSampler(pair::Pair{NodeName}...; max_nsamples::Integer=100) =
        RejectionSampler(Assignment(pair), max_nsamples)

function Random.rand!(a::Assignment, bn::BayesNet, sampler::RejectionSampler)
    for sample_count in 1 : sampler.max_nsamples
        rand!(a, bn)
        if consistent(a, sampler.evidence)
            return a
        end
    end
    return empty!(a)
end

Random.rand(bn::BayesNet, nsamples::Integer, evidence::Assignment) =
        rand(bn, RejectionSampler(evidence, 100), nsamples)

Random.rand(bn::BayesNet, nsamples::Integer, pair::Pair{NodeName}...) =
        rand(bn, nsamples, Assignment(pair))

#
# Likelihood Sampling
#
"""
Draw an assignment from the Bayesian network but set any variables in the evidence accordingly.
Returns the assignment and the probability weighting associated with the evidence.
"""
function get_weighted_sample!(a::Assignment, bn::BayesNet, evidence::Assignment)
    w = 1.0
    for cpd in bn.cpds
        varname = name(cpd)
        if haskey(evidence, varname)
            a[varname] = evidence[varname]
            w *= pdf(cpd, a)
        else
            a[varname] = rand(cpd, a)
        end
    end

    return (a, w)
end

get_weighted_sample!(a::Assignment, bn::BayesNet, pair::Pair{NodeName}...) =
        get_weighted_sample!(a, bn, Assignment(pair))

get_weighted_sample(bn::BayesNet, evidence::Assignment) =
        get_weighted_sample!(Assignment(), bn, evidence)

get_weighted_sample(bn::BayesNet, pair::Pair{NodeName}...) =
        get_weighted_sample(bn, Assignment(pair))

sample_weighted_dataframe(weighted_dataframe::DataFrame) =
        sample_weighted_dataframe!(Assignment(), weighted_dataframe)

"""
Likelihood Weighted Sampling
"""
struct LikelihoodWeightedSampler <: BayesNetSampler
    evidence::Assignment
end
LikelihoodWeightedSampler(pair::Pair{NodeName}...) =
        LikelihoodWeightedSampler(Assignment(pair))

function Random.rand!(a::Assignment, bn::BayesNet, sampler::LikelihoodWeightedSampler)
    get_weighted_sample!(a, sampler.weighted_dataframe)
    return a
end

Random.rand(bn::BayesNet, sampler::LikelihoodWeightedSampler, nsamples::Integer) =
        get_weighted_dataframe(bn, nsamples, sampler.evidence)

#
# only used by the src/gibbs code, not the inference code
#
"""
A dataset of variable assignments is obtained with an additional column
of weights in accordance with the likelihood of each assignment.
"""
function get_weighted_dataframe(bn::BayesNet, nsamples::Integer, evidence::Assignment)

    t = Dict{Symbol, Vector{Any}}()
    for name in names(bn)
        t[name] = Any[]
    end

    w = ones(Float64, nsamples)
    a = Assignment()

    for i in 1:nsamples
        a, weight = get_weighted_sample!(a, bn, evidence)
        for (varname, val) in a
             push!(t[varname], val)
        end
        w[i] = weight
    end
    t[:p] = w / sum(w)

    convert(DataFrame, t)
end

get_weighted_dataframe(bn::BayesNet, nsamples::Integer, pair::Pair{NodeName}...) =
        get_weighted_dataframe(bn, nsamples, Assignment(pair))

"""
Chooses a sample at random from a weighted dataframe
"""
function sample_weighted_dataframe!(a::Assignment, weighted_dataframe::DataFrame)
    p = weighted_dataframe[:, :p]
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end
    for varname in names(weighted_dataframe)
        if varname != :p
            a[varname] = weighted_dataframe[i, varname]
        end
    end
    return a
end
