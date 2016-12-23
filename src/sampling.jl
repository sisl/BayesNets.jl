"""
Absract type for sampling with Base.rand(BayesNet, BayesNetSampler, nsamples)
                               Base.rand!(Assignemnt, BayesNet, BayesNetSampler)
                               Base.rand(BayesNet, BayesNetSampler)
"""
abstract BayesNetSampler

"""
Overwrites assignment with a sample from bn using the sampler
"""
Base.rand!(a::Assignment, bn::BayesNet, sampler::BayesNetSampler) = error("rand! not implemented for $(typeof(sampler))")

"""
Returns an assignment sampled from the bn using the provided sampler
"""
Base.rand(bn::BayesNet, sampler::BayesNetSampler) = rand!(Assignment(), bn, sampler)

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows.
"""
function Base.rand(bn::BayesNet, sampler::BayesNetSampler, nsamples::Integer)

    a = rand(bn, sampler)
    df = DataFrame()
    for cpd in bn.cpds
        df[name(cpd)] = Array(typeof(a[name(cpd)]), nsamples)
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


"""
Straightforward sampling from a BayesNet.
The default sampler.
"""
type DirectSampler <: BayesNetSampler end
function Base.rand!(a::Assignment, bn::BayesNet, sampler::DirectSampler)
    for cpd in bn.cpds
        a[name(cpd)] = rand(cpd, a)
    end
    a
end
Base.rand!(a::Assignment, bn::BayesNet) = rand!(a, bn, DirectSampler())
Base.rand(bn::BayesNet, nsamples::Integer) = rand(bn, DirectSampler(), nsamples)
Base.rand(bn::BayesNet) = rand(bn, DirectSampler())


"""
Rejection Sampling in which the assignments are forced to be consistent with the provided values.
Each sampler is attempted at most `max_nsamples` times before returning an empty assignment.
"""
type RejectionSampler <: BayesNetSampler
    evidence::Assignment
    max_nsamples::Int
end
RejectionSampler(pair::Pair{NodeName}...; max_nsamples::Integer=100) = RejectionSampler(Assignment(pair), max_nsamples)
function Base.rand!(a::Assignment, bn::BayesNet, sampler::RejectionSampler)
    for sample_count in 1 : sampler.max_nsamples
        rand!(a, bn)
        if consistent(a, sampler.evidence)
            return a
        end
    end
    return empty!(a)
end
Base.rand(bn::BayesNet, nsamples::Integer, evidence::Assignment) = rand(bn, RejectionSampler(evidence, 100), nsamples)
Base.rand(bn::BayesNet, nsamples::Integer, pair::Pair{NodeName}...) = rand(bn, nsamples, Assignment(pair))

"""
Weighted Sampling in which a dataset of variable assignments is obtained with an additional column
of weights in accordance with the likelihood of each assignment.

Unlike other samplers, this one only supports rand(bn, sampler, nsamples)
"""
type WeightedSampler <: BayesNetSampler
    evidence::Assignment
end
WeightedSampler(pair::Pair{NodeName}...) = WeightedSampler(Assignment(pair))
function Base.rand(bn::BayesNet, sampler::WeightedSampler, nsamples::Integer)

    t = Dict{Symbol, Vector{Any}}()
    for name in names(bn)
        t[name] = Any[]
    end

    w = ones(Float64, nsamples)
    a = Assignment()

    for i in 1:nsamples
        for cpd in bn.cpds
            varname = name(cpd)
            if haskey(sampler.evidence, varname)
                a[varname] = sampler.evidence[varname]
                w[i] *= pdf(cpd, a)
            else
                a[varname] = rand(cpd, a)
            end
            push!(t[varname], a[varname])
        end
    end
    t[:p] = w / sum(w)

    convert(DataFrame, t)
end

"""
Chooses a sample at random from the result of a WeightedSampler sampling
"""
function sample_weighted_dataframe(rand_samples::DataFrame)
    p = rand_samples[:, :p]
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end
    a = Assignment()
    for varname in names(rand_samples)
        if varname != :p
            a[varname] = rand_samples[i, varname]
        end
    end
    return a
end

