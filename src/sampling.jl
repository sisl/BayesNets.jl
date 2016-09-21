"""
Overwrites assignment with a joint sample from bn
    NOTE: this will condition as it goes
"""
function Base.rand!(a::Assignment, bn::BayesNet)
    for cpd in bn.cpds
        a[name(cpd)] = rand(cpd, a)
    end
    a
end
Base.rand(bn::BayesNet) = rand!(Assignment(), bn)

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows.
"""
function Base.rand(bn::BayesNet, nsamples::Integer)

    a = rand(bn)
    df = DataFrame()
    for cpd in bn.cpds
        df[name(cpd)] = Array(typeof(a[name(cpd)]), nsamples)
    end

    for i in 1:nsamples
        rand!(a, bn)
        for cpd in bn.cpds
            n = name(cpd)
            df[i, n] = a[n]
        end
    end

    df
end

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows or errors out

nsamples: the number of rows the resulting DataFrame will contain
consistent_with: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1)
max_nsamples: an upper limit on the number of samples that will be tried, needed to ensure zero-prob samples are never used
"""
function Base.rand(bn::BayesNet, nsamples::Integer, consistent_with::Assignment; max_nsamples::Integer=nsamples*100)

    a = rand(bn)
    df = DataFrame()
    for cpd in bn.cpds
        df[name(cpd)] = Array(typeof(a[name(cpd)]), nsamples)
    end

    sample_count = 0
    for i in 1:nsamples

        while sample_count ≤ max_nsamples

            rand!(a, bn)
            if consistent(a, consistent_with)
                break
            end

            sample_count += 1
        end

        sample_count ≤ max_nsamples || error("rand hit sample threshold of $max_nsamples")

        for cpd in bn.cpds
            n = name(cpd)
            df[i, n] = a[n]
        end
    end

    df
end
function Base.rand(bn::BayesNet, nsamples::Integer, pair::Pair{NodeName}...; max_nsamples::Integer=nsamples*100)
    a = Assignment(pair)
    rand(bn, nsamples, a, max_nsamples=max_nsamples)
end

"""
    rand_table_weighted(bn::BayesNet; nsamples::Integer=10, consistent_with::Assignment=Assignment())
Generates a DataFrame containing a dataset of variable assignments using weighted sampling.
Always return a DataFrame with `nsamples` rows.
"""
function rand_table_weighted(bn::BayesNet; nsamples::Integer=10, consistent_with::Assignment=Assignment())

    t = Dict(name => Any[] for name in names(bn))
    w = ones(Float64, nsamples)
    a = Assignment()

    for i in 1:nsamples
        for cpd in bn.cpds
            varname = name(cpd)
            if haskey(consistent_with, varname)
                a[varname] = consistent_with[varname]
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