"""
Abstract type for sampling with Base.rand(ProbabilisticGraphicalModel, Sampler, nsamples)
                                Base.rand!(Assignment, ProbabilisticGraphicalModel, Sampler)
                                Base.rand(ProbabilisticGraphicalModel, Sampler)
"""
abstract type Sampler end

"""
Overwrites Assignment with a sample from the PGM using the given Sampler
"""
Random.rand!(a::Assignment, pgm::ProbabilisticGraphicalModel, sampler::Sampler) = error("rand! not implemented for $(typeof(Sampler))")

"""
Returns a new Assignment sampled from the PGM using the provided sampler
"""
Random.rand(pgm::ProbabilisticGraphicalModel, sampler::Sampler) = rand!(Assignment(), bn, sampler)

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows.
"""
function Random.rand(pgm::ProbabilisticGraphicalModel, sampler::Sampler, nsamples::Integer)

    a = rand(pgm, sampler)
    df = DataFrame()
    nodenames = names(pgm)
    for nodename in nodenames
        df[nodename] = Array{typeof(a[nodename])}(nsamples)
    end

    for i in 1:nsamples
        rand!(a, pgm, sampler)
        for name in nodenames
            df[i, name] = a[name]
        end
    end

    df
end
