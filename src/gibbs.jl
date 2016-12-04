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

"""
Technically modifies gss because gss stores the cache for this function
"""
function get_markov_blanket_cpds(gss::GibbsSamplerState, varname::Symbol)
    if haskey(gss.markov_blanket_cache, varname)
        return gss.markov_blanket_cache[varname]
    end

    _markov_blanket_cpds = markov_blanket_cpds(gss.bn, varname)
    gss.markov_blanket_cache[varname] = _markov_blanket_cpds
    return _markov_blanket_cpds
end

"""
Modifies a and gss
"""
function get_finite_distribution!(gss::GibbsSamplerState, varname::Symbol, a::Assignment, support::AbstractArray)
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

    markov_blanket_cpds = get_markov_blanket_cpds(gss, varname)
    posterior_distribution = zeros(length(support))
    for (index, domain_element) in enumerate(support)
        a[varname] = domain_element
        # Sum logs for numerical stability
        posterior_distribution[index] = exp(sum([logpdf(cpd, a) for cpd in markov_blanket_cpds]))
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

"""
set a[varname] ~ P(varname | not varname)

Modifies both a and gss
"""
function sample_posterior_finite!(gss::GibbsSamplerState, varname::Symbol, a::Assignment, support::AbstractArray)

   posterior_distribution = get_finite_distribution!(gss, varname, a, support)

   # Adapted from Distributions.jl, credit to its authors
   p = posterior_distribution
   n = length(p)
   i = 1
   c = p[1]
   u = rand()
   while c < u && i < n
       c += p[i += 1]
   end

   a[varname] = support[i]

end

"""
Implements Metropolis-Hastings with a normal distribution proposal with mean equal to the previous value
of the variable "varname" and stddev equal to 10 times the standard deviation of the distribution of the target
variable given its parents ( var_distribution should be get(bn, varname)(a) )

MH will go through nsamples iterations.  If no proposal is accepted, the original value will remain

This function expects that a[varname] is within the support of the distribution, it will not check to make sure this is true

set a[varname] ~ P(varname | not varname)

Modifies a and caches in gss
"""
function sample_posterior_continuous!(gss::GibbsSamplerState, varname::Symbol, a::Assignment, 
                                     var_distribution::ContinuousUnivariateDistribution; MH_iterations::Integer=10)
    # TODO consider using slice sampling or having an option for slice sampling
    # Implement http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7080917

    # Random Walk Metropolis Hastings
    markov_blanket_cpds = get_markov_blanket_cpds(gss, varname)
    # TODO What should this stddev be?  The product of the stddev of all cpds in the markov blanket?
    # TODO consider using a TruncatedNormal(mu::Real, sigma::Real, a::Real, b::Real) when the support is bounded on either or both sides
    stddev = std(var_distribution) * 10.0
    previous_sample_scaled_true_prob = exp(sum([logpdf(cpd, a) for cpd in markov_blanket_cpds]))
    proposal_distribution = Normal(a[varname], stddev) # TODO why does calling this constructor take so long?

    for sample_iter = 1:MH_iterations

        # compute proposed jump
        current_value = a[varname]
        proposed_jump = rand(proposal_distribution)
        if ~ insupport(var_distribution, proposed_jump)
            continue # reject immediately, zero probability
        end

        # Compute acceptance probability
        a[varname] = proposed_jump
        proposed_jump_scaled_true_prob = exp(sum([logpdf(cpd, a) for cpd in markov_blanket_cpds]))
        # Our proposal is symmetric, so q(X_new, X_old) / q(X_old, X_new) = 1
        # accept_prob = min(1, proposed_jump_scaled_true_prob/previous_sample_scaled_true_prob)
        accept_prob = proposed_jump_scaled_true_prob/previous_sample_scaled_true_prob # min operation is unnecessary

        # Accept or reject and clean up
        if rand() < accept_prob
            # a[varname] = proposed_jump
            previous_sample_scaled_true_prob = proposed_jump_scaled_true_prob
            proposal_distribution = Normal(proposed_jump, stddev)
        else
            a[varname] = current_value
        end
    end

    # a[varname] is set in the above for loop

end

"""
set a[varname] ~ P(varname | not varname)

Modifies a and caches in gss
"""
function sample_posterior!(gss::GibbsSamplerState, varname::Symbol, a::Assignment)

    bn = gss.bn
    cpd = get(bn, varname)
    distribution = cpd(a)
    if hasfinitesupport(distribution)
        sample_posterior_finite!(gss, varname, a, support(distribution))
    elseif typeof(distribution) <: DiscreteUnivariateDistribution
        error("Infinite Discrete distributions are currently not supported in the Gibbs sampler")
    else
        sample_posterior_continuous!(gss, varname, a, distribution)
    end
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
                 sample_posterior!(gss, varname, a)
            end

            if isnull(variable_order)
                v_order = shuffle!(v_order)
            end
        end

        for varname in v_order
            sample_posterior!(gss, varname, a)
            push!(t[varname], a[varname])
        end

    end

    return convert(DataFrame, t), Integer(now() - start_time)
end

"""
TODO description

The Gibbs Sampler only supports CPDs that return Univariate Distributions CDP{D<:UnivariateDistribution}

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