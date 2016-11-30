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

    t = Dict{Symbol, Vector{Any}}()
    for name in names(bn)
        t[name] = Any[]
    end

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

type GibbsSamplerState

    bn::BayesNet
    name_order::Array{Symbol,1}
    max_cache_size::Nullable{Integer}
    markov_blanket_cache::Dict{Symbol, Array{CPD}}
    finite_distribution_cache::Dict{String, Array{Float64, 1}}

    function GibbsSamplerState(
        bn::BayesNet,
        max_cache_size::Nullable{Integer}=Nullable{Integer}()
        )

        new(bn, names(bn), max_cache_size, Dict{Symbol, Array{CPD}}(), Dict{String, Array{Float64, 1}}())
    end

end

function get_markov_blanket_cpds(gss::GibbsSamplerState, varname::Symbol)
    if haskey(gss.markov_blanket_cache, varname)
        return gss.markov_blanket_cache[varname]
    end

    bn = gss.bn
    markov_blanket_cdps = [get(bn, child_name) for child_name in children(bn, varname)]
    markov_blanket_cdps = convert(Array{CPD}, markov_blanket_cdps) # Make type explicit or next line will fail
    push!(markov_blanket_cdps, get(bn, varname))
    gss.markov_blanket_cache[varname] = markov_blanket_cdps
    return markov_blanket_cdps
end

# Modifies a
function get_finite_distribution(gss::GibbsSamplerState, varname::Symbol, a::Assignment, support::AbstractArray)
    a[varname] = varname
    # Best way to compute this key?
    # A quick test showed that this method was faster than
    # using Array{String, 1} (no join)
    # using Array{Any, 1} (no stringification)
    # using a tuple
    # not caching at all
    key = join([string(a[name]) for name in gss.name_order], ",")

    if haskey(gss.finite_distribution_cache, key)
        return gss.finite_distribution_cache[key]
    end

    markov_blanket_cdps = get_markov_blanket_cpds(gss, varname)
    posterior_distribution = zeros(length(support))
    for (index, domain_element) in enumerate(support)
        a[varname] = domain_element
        # Sum logs for numerical stability
        posterior_distribution[index] = exp(sum([logpdf(cdp, a) for cdp in markov_blanket_cdps]))
    end
    posterior_distribution = posterior_distribution / sum(posterior_distribution)
    if isnull(gss.max_cache_size) || length(gss.finite_distribution_cache) < get(gss.max_cache_size)
        gss.finite_distribution_cache[key] = posterior_distribution
    end
    return posterior_distribution
end

function sample_weighted_dataframe(rand_samples::DataFrame)
    p = rand_samples[:, :p]
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end

    return Assignment(Dict(varname => rand_samples[i, varname] for varname in names(rand_samples) if varname != :p))
end

function sample_posterior_finite(gss::GibbsSamplerState, varname::Symbol, a::Assignment, support::AbstractArray)

   posterior_distribution = get_finite_distribution(gss, varname, a, support)

   # Adapted from Distributions.jl, credit to its authors
   p = posterior_distribution
   n = length(p)
   i = 1
   c = p[1]
   u = rand()
   while c < u && i < n
       c += p[i += 1]
   end
   return support[i]

end

function sample_posterior_continuous(gss::GibbsSamplerState, varname::Symbol, a::Assignment; nsamples::Integer=20)
    # TODO likelihood weighted sampling may not be the correct way to do this
    # TODO if likelihood weighted sampling is used, then you must keep sampling until a sample with non-zero probability occurs
    # TODO if likelihood weighted sampling is used, then make the nsamples parameter here a parameter in gibbs_sample
    # TODO consider using slice sampling or having an option for slice sampling

    # Implement http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7080917
    # TODO implement MH with the proposal being a normal distribution centered on the previous example with large std dev
    bn = gss.bn

    children_cdps = [get(bn, child_name) for child_name in children(bn, varname)]
    var_cpd = get(bn, varname)

    # TODO if this doesn't get replaced with another method (it likely will) then preallocate everything properly
    t = Dict{Symbol, Vector{Any}}()
    t[varname] = Any[]

    w = ones(Float64, nsamples)

    for i in 1:nsamples
        a[varname] = rand(var_cpd, a)
        for cpd in children_cdps
            w[i] *= pdf(cpd, a)
        end
        push!(t[varname], a[varname])
    end
    t[:p] = w / sum(w)
    t = convert(DataFrame, t)

    assignment = sample_weighted_dataframe(t)
    return assignment[varname]
end

"""
Temporarily modifies a, but restores it after computations
"""
function sample_posterior(gss::GibbsSamplerState, varname::Symbol, a::Assignment)
    original_value = a[varname]

    bn = gss.bn
    cpd = get(bn, varname)
    distribution = cpd(a)
    if hasfinitesupport(distribution)
        new_value = sample_posterior_finite(gss, varname, a, support(distribution))
    elseif typeof(distribution) <: DiscreteUnivariateDistribution
        error("Infinite Discrete distributions are currently not supported in the Gibbs sampler")
    else
        new_value = sample_posterior_continuous(gss, varname, a)
    end

    a[varname] = original_value
    return new_value
end

function gibbs_sample_main_loop(gss::GibbsSamplerState, nsamples::Integer, sample_skip::Integer, 
start_sample::Assignment, consistent_with::Assignment, variable_order::Nullable{Vector{Symbol}},
time_limit::Nullable{Integer})

    start_time = now()

    bn = gss.bn
    a = start_sample
    if isnull(variable_order)
         v_order = names(bn)
    else
         v_order = get(variable_order)
    end

    v_order = [varname for varname in v_order if ~haskey(consistent_with, varname)]

    t = Dict{Symbol, Vector{Any}}()
    for name in v_order
        t[name] = Any[]
    end

    for sample_iter in 1:nsamples
        if (~ isnull(time_limit)) && (Integer(now() - start_time) > get(time_limit))
            break
        end

        if isnull(variable_order)
            v_order = shuffle!(v_order)
        end

        # skip over sample_skip samples
        for skip_iter in 1:sample_skip
            for varname in v_order
                 a[varname] = sample_posterior(gss, varname, a)
            end

            if isnull(variable_order)
                v_order = shuffle!(v_order)
            end
        end

        for varname in v_order
            a[varname] = sample_posterior(gss, varname, a)
            push!(t[varname], a[varname])
        end

    end

    return convert(DataFrame, t), Integer(now() - start_time)
end

"""
TODO description

First burn_in samples will be sampled and then discrded.  Next additional samples will be draw, 
and every (sample_skip + 1)th sample will be returned while the rest are discarded.

The algorithm will return the samples it has collected when either nsamples samples have been collected or time_limit milliseconds have passed.  If time_limit is not specified then the algorithm will run until nsamples have been collected.

sample_skip: every (sample_skip + 1)th sample will be used, the other sample_skip samples will be thrown out.  The higher the sample_skip, the less correlated samples will be but the longer the computation time per sample.
variable_order: variable_order determines the order of variables changed when generating a new sample, if variable_order is None, then a random variable order will be used for each sample.

max_cache_size:  If null, cache as much as possible, otherwise cache at most "max_cache_size"  distributions
"""
function gibbs_sample(bn::BayesNet, nsamples::Integer, burn_in::Integer; sample_skip::Integer=99,
consistent_with::Assignment=Assignment(), variable_order::Nullable{Vector{Symbol}}=Nullable{Vector{Symbol}}(), 
time_limit::Nullable{Integer}=Nullable{Integer}(), error_if_time_out::Bool=true, 
initial_sample::Nullable{Assignment}=Nullable{Assignment}(), max_cache_size::Nullable{Integer}=Nullable{Integer}())
    """
    TODO come up with an automatic method for setting the burn_in period, look at literature.  
              Once this is implemented, move burn_in to the default parameters
    TODO rename sample_skip to thinning
    """
    # check parameters for correctness
    nsamples > 0 || throw(ArgumentError("nsamples parameter less than 1"))
    burn_in >= 0 || throw(ArgumentError("Negative burn_in parameter"))
    sample_skip >= 0 || throw(ArgumentError("Negative sample_skip parameter"))
    if ~ isnull(variable_order)
        v_order = get(variable_order)
        bn_names = names(bn)
        for name in bn_names
            name in v_order || throw(ArgumentError("Gibbs sample variable_order must contain all variables in the Bayes Net"))
        end
        for name in v_order
            name in bn_names || throw(ArgumentError("Gibbs sample variable_order contains a variable not in the Bayes Net"))
        end
    end
    if ~ isnull(time_limit)
        get(time_limit) > 0 || throw(ArgumentError(join(["Invalid time_limit specified (", get(time_limit), ")"])))
    end
    if ~ isnull(initial_sample)
        init_sample = get(initial_sample)
        for name in names(bn)
            haskey(init_sample, name) || throw(ArgumentError("Gibbs sample initial_sample must be an assignment with all variables in the Bayes Net"))
        end
        for name in keys(consistent_with)
            init_sample[name] == consistent_with[name] || throw(ArgumentError("Gibbs sample initial_sample was inconsistent with consistent_with"))
        end
    end

    gss = GibbsSamplerState(bn, max_cache_size)
   
    # Burn in 
    # for burn_in_initial_sample use rand_table_weighted, should be consistent with the varibale consistent_with
    if isnull(initial_sample)
        rand_samples = rand_table_weighted(bn, nsamples=10, consistent_with=consistent_with)
	if any(isnan(convert(Array{AbstractFloat}, rand_samples[:p])))
		error("Gibbs Sampler was unable to find an inital sample with non-zero probability")
	end
        burn_in_initial_sample = sample_weighted_dataframe(rand_samples)
    else
        burn_in_initial_sample = get(initial_sample)
    end
    burn_in_samples, burn_in_time = gibbs_sample_main_loop(gss, burn_in, 0, burn_in_initial_sample, 
                                         consistent_with, variable_order, time_limit)
    remaining_time = Nullable{Integer}()
    if ~isnull(time_limit)
        remaining_time = Nullable{Integer}(get(time_limit) - burn_in_time)
        if error_if_time_out
            get(remaining_time) > 0 || error("Time expired during Gibbs sampling")
        end
    end
   
    # Real samples
    main_samples_initial_sample = burn_in_initial_sample
    if burn_in != 0 && size(burn_in_samples)[1] > 0
        main_samples_initial_sample = Assignment(Dict(varname => 
                      (haskey(consistent_with, varname) ? consistent_with[varname] : burn_in_samples[end, varname])
                      for varname in names(bn))) 
    end
    samples, samples_time = gibbs_sample_main_loop(gss, nsamples, sample_skip, 
                               main_samples_initial_sample, consistent_with, variable_order, remaining_time)
    combined_time = burn_in_time + samples_time
    if error_if_time_out && ~isnull(time_limit)
        combined_time < get(time_limit) || error("Time expired during Gibbs sampling")
    end

    # Add in columns for variables that were conditioned on
    evidence = DataFrame(Dict(varname => ones(size(samples)[1]) * consistent_with[varname] 
                 for varname in keys(consistent_with)))
    return hcat(samples, evidence)
end
