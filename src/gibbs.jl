"""
Used to cache various things the Gibbs sampler needs
"""
mutable struct GibbsSamplerState

    bn::BayesNet
    key_constructor_name_order::Array{Symbol,1}
    max_cache_size::Union{Int, Nothing}
    markov_blanket_cpds_cache::Dict{Symbol, Array{CPD}}
    markov_blanket_cache::Dict{Symbol, Vector{Symbol}}
    finite_distrbution_is_cacheable::Dict{Symbol, Bool}
    finite_distribution_cache::Dict{String, Array{Float64, 1}}

    function GibbsSamplerState(
        bn::BayesNet,
        max_cache_size::Union{Int, Nothing}=nothing
        )

        a = rand(bn)
        markov_blankets = Dict{Symbol, Vector{Symbol}}()
        is_cacheable = Dict{Symbol, Bool}()
        markov_blanket_cpds_cache = Dict{Symbol, Array{CPD}}()
        for name in names(bn)
            markov_blankets[name] = Symbol[ele for ele in markov_blanket(bn, name)]
            is_cacheable[name] = reduce(&, [hasfinitesupport(get(bn, mb_name)(a)) for mb_name in markov_blankets[name]])
            markov_blanket_cpds_cache[name] = get(bn, markov_blanket(bn, name))
        end

        new(bn, names(bn), max_cache_size,
            markov_blanket_cpds_cache,
            markov_blankets,
            is_cacheable,
            Dict{String, Array{Float64, 1}}(),
            )
    end
end

"""
Helper to sample_posterior_finite

Modifies a and gss
"""
function get_finite_distribution!(gss::GibbsSamplerState, varname::NodeName, a::Assignment, support::AbstractArray)

    is_cacheable = gss.finite_distrbution_is_cacheable[varname]
    key = ""
    if is_cacheable

        key = join([string(a[name]) for name in gss.markov_blanket_cache[varname]], ",")
        key = join([string(varname), key], ",")

        if haskey(gss.finite_distribution_cache, key)
            return gss.finite_distribution_cache[key]
        end

    end

    markov_blanket_cpds = gss.markov_blanket_cpds_cache[varname]
    posterior_distribution = zeros(length(support))
    for (index, domain_element) in enumerate(support)
        a[varname] = domain_element
        # Sum logs for numerical stability
        posterior_distribution[index] = exp(sum([logpdf(cpd, a) for cpd in markov_blanket_cpds]) +
                                                 logpdf(get(gss.bn, varname), a)) # because A is not in its own Markov blanket
    end
    posterior_distribution = posterior_distribution / sum(posterior_distribution)
    if is_cacheable && ( gss.max_cache_size == nothing || length(gss.finite_distribution_cache) < gss.max_cache_size )
        gss.finite_distribution_cache[key] = posterior_distribution
    end
    return posterior_distribution
end

"""
Helper to sample_posterior
Should only be called if the variable associated with varname is discrete

set a[varname] ~ P(varname | not varname)

Modifies both a and gss
"""
function sample_posterior_finite!(gss::GibbsSamplerState, name::NodeName, a::Assignment, support::AbstractArray)

   posterior_distribution = get_finite_distribution!(gss, name, a, support)

   # Adapted from Distributions.jl, credit to its authors
   p = posterior_distribution
   n = length(p)
   i = 1
   c = p[1]
   u = rand()
   while c < u && i < n
       c += p[i += 1]
   end

   a[name] = support[i]
end

"""
Implements Metropolis-Hastings with a normal distribution proposal with mean equal to the previous value
of the variable "varname" and stddev equal to 10 times the standard deviation of the distribution of the target
variable given its parents ( var_distribution should be get(bn, varname)(a) )

MH will go through nsamples iterations.  If no proposal is accepted, the original value will remain

This function expects that a[varname] is within the support of the distribution, it will not check to make sure this is true

Helper to sample_posterior
Should only be used to sampling continuous distributions

set a[varname] ~ P(varname | not varname)

Modifies a and caches in gss
"""
function sample_posterior_continuous!(
    gss::GibbsSamplerState,
    varname::NodeName,
    a::Assignment,
    var_distribution::ContinuousUnivariateDistribution;
    MH_iterations::Integer=10,
    )

    # Random Walk Metropolis Hastings
    markov_blanket_cpds = gss.markov_blanket_cpds_cache[varname]
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
        if rand() <= accept_prob
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
        # sample_posterior_continuous_adaptive!(gss, varname, a, distribution)
        sample_posterior_continuous!(gss, varname, a, distribution)
    end
end

"""
The main loop associated with Gibbs sampling
Returns a data frame with nsamples samples

Supports the various parameters supported by gibbs_sample
Refer to gibbs_sample for parameter meanings
"""
function gibbs_sample_main_loop(
    gss::GibbsSamplerState,
    nsamples::Integer,
    thinning::Integer,
    start_sample::Assignment,
    consistent_with::Assignment,
    variable_order::Union{Vector{Symbol}, Nothing},
    time_limit::Union{Int, Nothing},
    )

    start_time = now()

    bn = gss.bn
    a = start_sample
    if variable_order == nothing
         v_order = names(bn)
    else
         v_order = coalesce(variable_order)
    end

    # v_order = [varname for varname in v_order if ~haskey(consistent_with, varname)]
    v_order_query_only = Symbol[]
    for varname in v_order
        if ~haskey(consistent_with, varname)
            push!(v_order_query_only, varname)
        end
    end
    v_order = v_order_query_only

    t = Dict{Symbol, Vector{Any}}()
    for name in v_order
        t[name] = Any[]
    end

    for sample_iter in 1:nsamples
        if (time_limit!=nothing) && ((now() - start_time).value > time_limit)
            break
        end

        if variable_order == nothing
            v_order = shuffle!(v_order)
        end

        # skip over thinning samples
        for skip_iter in 1:thinning
            for varname in v_order
                 sample_posterior!(gss, varname, a)
            end

            if variable_order == nothing
                v_order = shuffle!(v_order)
            end
        end

        for varname in v_order
            sample_posterior!(gss, varname, a)
            push!(t[varname], a[varname])
        end

    end

    return convert(DataFrame, t), (now() - start_time).value
end

"""
Implements Gibbs sampling. (https://en.wikipedia.org/wiki/Gibbs_sampling)
For finite variables, the posterior distribution is sampled by building the exact distribution.
For continuous variables, the posterior distribution is sampled using Metropolis Hastings MCMC.
Discrete variables with infinite support are currently not supported.
The Gibbs Sampler only supports CPDs that return Univariate Distributions. (CPD{D<:UnivariateDistribution})

bn:: A Bayesian Network to sample from.  bn should only contain CPDs that return UnivariateDistributions.

nsamples: The number of samples to return.

burn_in:  The first burn_in samples will be discarded.  They will not be returned.
The thinning parameter does not affect the burn in period.
This is used to ensure that the Gibbs sampler converges to the target stationary distribution before actual samples are drawn.

thinning: For every thinning + 1 number of samples drawn, only the last is kept.
Thinning is used to reduce autocorrelation between samples.
Thinning is not used during the burn in period.
e.g. If thinning is 1, samples will be drawn in groups of two and only the second sample will be in the output.

time_limit: The number of milliseconds to run the algorithm.
The algorithm will return the samples it has collected when either nsamples samples have been collected or time_limit
milliseconds have passed.  If time_limit is null then the algorithm will run until nsamples have been collected.
This means it is possible that zero samples are returned.

error_if_time_out: If error_if_time_out is true and the time_limit expires, an error will be raised.
If error_if_time_out is false and the time limit expires, the samples that have been collected so far will be returned.
	This means it is possible that zero samples are returned.  Burn in samples will not be returned.
If time_limit is null, this parameter does nothing.

consistent_with: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1).
Use to sample conditional distributions.

max_cache_size:  If null, cache as much as possible, otherwise cache at most "max_cache_size"  distributions

variable_order: variable_order determines the order of variables changed when generating a new sample.
If null use a random order for every sample (this is different from updating the variables at random).
Otherwise should be a list containing all the variables in the order they should be updated.

initial_sample:  The inital assignment to variables to use.  If null, the initial sample is chosen by
briefly running rand(bn, get_weighted_dataframe).
"""
function gibbs_sample(bn::BayesNet, nsamples::Integer, burn_in::Integer;
        thinning::Integer=0,
        consistent_with::Assignment=Assignment(),
        variable_order::Union{Vector{Symbol}, Nothing}=nothing,
        time_limit::Union{Int, Nothing}=nothing,
        error_if_time_out::Bool=true,
        initial_sample::Union{Assignment, Nothing}=nothing,
        max_cache_size::Union{Int, Nothing}=nothing
        )
    # check parameters for correctness
    nsamples > 0 || throw(ArgumentError("nsamples parameter less than 1"))
    burn_in >= 0 || throw(ArgumentError("Negative burn_in parameter"))
    thinning >= 0 || throw(ArgumentError("Negative thinning parameter"))
    if variable_order != nothing
        v_order = coalesce(variable_order)
        bn_names = names(bn)
        for name in bn_names
            name in v_order || throw(ArgumentError("Gibbs sample variable_order must contain all variables in the Bayes Net"))
        end
        for name in v_order
            name in bn_names || throw(ArgumentError("Gibbs sample variable_order contains a variable not in the Bayes Net"))
        end
    end
    if time_limit != nothing
        time_limit > 0 || throw(ArgumentError(join(["Invalid time_limit specified (", time_limit, ")"])))
    end
    if initial_sample != nothing
        init_sample = coalesce(initial_sample)
        for name in names(bn)
            haskey(init_sample, name) || throw(ArgumentError("Gibbs sample initial_sample must be an assignment with all variables in the Bayes Net"))
        end
        for name in keys(consistent_with)
            init_sample[name] == consistent_with[name] || throw(ArgumentError("Gibbs sample initial_sample was inconsistent with consistent_with"))
        end
        pdf(bn, init_sample) > 0 || throw(ArgumentError("Gibbs sample initial_sample has a pdf value of zero"))
    end

    gss = GibbsSamplerState(bn, max_cache_size)

    # Burn in
    # for burn_in_initial_sample use get_weighted_dataframe, should be consistent with the varibale consistent_with
    if initial_sample == nothing
        rand_samples = get_weighted_dataframe(bn, 50, consistent_with)
    	if reduce(|, isnan.(convert(Array{AbstractFloat}, rand_samples[:p])))
    		error("Gibbs Sampler was unable to find an inital sample with non-zero probability, please supply an inital sample")
    	end
        burn_in_initial_sample = sample_weighted_dataframe(rand_samples)
    else
        burn_in_initial_sample = initial_sample
    end
    burn_in_samples, burn_in_time = gibbs_sample_main_loop(gss, burn_in, 0, burn_in_initial_sample,
                                         consistent_with, variable_order, time_limit)

    # Check that more time is available
    remaining_time = nothing
    if time_limit != nothing
        remaining_time = time_limit - burn_in_time
        if error_if_time_out
            remaining_time > 0 || error("Time expired during Gibbs sampling")
        end
    end

    # Real samples
    main_samples_initial_sample = burn_in_initial_sample
    if burn_in != 0 && size(burn_in_samples)[1] > 0
        main_samples_initial_sample = Assignment()
        for varname in names(bn)
            main_samples_initial_sample[varname] = haskey(consistent_with, varname) ?
                                                       consistent_with[varname] : burn_in_samples[end, varname]
        end
    end
    samples, samples_time = gibbs_sample_main_loop(gss, nsamples, thinning,
                               main_samples_initial_sample, consistent_with, variable_order, remaining_time)
    combined_time = burn_in_time + samples_time
    if error_if_time_out && time_limit!=nothing
        combined_time < time_limit || error("Time expired during Gibbs sampling")
    end

    # Add in columns for variables that were conditioned on
    evidence = Dict{Symbol, Array{Float64,1}}()
    for varname in keys(consistent_with)
        evidence[varname] = ones(size(samples)[1]) * consistent_with[varname]
    end
    evidence = DataFrame(evidence)
    return hcat(samples, evidence)
end

"""
The GibbsSampler type houses the parameters of the Gibbs sampling algorithm.  The parameters are defined below:

burn_in:  The first burn_in samples will be discarded.  They will not be returned.
The thinning parameter does not affect the burn in period.
This is used to ensure that the Gibbs sampler converges to the target stationary distribution before actual samples are drawn.

thinning: For every thinning + 1 number of samples drawn, only the last is kept.
Thinning is used to reduce autocorrelation between samples.
Thinning is not used during the burn in period.
e.g. If thinning is 1, samples will be drawn in groups of two and only the second sample will be in the output.

time_limit: The number of milliseconds to run the algorithm.
The algorithm will return the samples it has collected when either nsamples samples have been collected or time_limit
milliseconds have passed.  If time_limit is null then the algorithm will run until nsamples have been collected.
This means it is possible that zero samples are returned.

error_if_time_out: If error_if_time_out is true and the time_limit expires, an error will be raised.
If error_if_time_out is false and the time limit expires, the samples that have been collected so far will be returned.
        This means it is possible that zero samples are returned.  Burn in samples will not be returned.
If time_limit is null, this parameter does nothing.

consistent_with: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1).
Use to sample conditional distributions.

max_cache_size:  If null, cache as much as possible, otherwise cache at most "max_cache_size"  distributions

variable_order: variable_order determines the order of variables changed when generating a new sample.
If null use a random order for every sample (this is different from updating the variables at random).
Otherwise should be a list containing all the variables in the order they should be updated.

initial_sample:  The inital assignment to variables to use.  If null, the initial sample is chosen by
briefly using a LikelihoodWeightedSampler.
"""
mutable struct GibbsSampler <: BayesNetSampler

    evidence::Assignment
    burn_in::Int
    thinning::Int
    variable_order::Union{Vector{Symbol}, Nothing}
    time_limit::Union{Int, Nothing}
    error_if_time_out::Bool
    initial_sample::Union{Assignment, Nothing}
    max_cache_size::Union{Int, Nothing}

    function GibbsSampler(evidence::Assignment=Assignment();
        burn_in::Int=100,
        thinning::Int=0,
        variable_order::Union{Vector{Symbol}, Nothing}=nothing,
        time_limit::Union{Int, Nothing}=nothing,
        error_if_time_out::Bool=true,
        initial_sample::Union{Assignment, Nothing}=nothing,
        max_cache_size::Union{Int, Nothing}=nothing
        )

        new(evidence, burn_in, thinning, variable_order, time_limit, error_if_time_out, initial_sample, max_cache_size)
    end
end

"""
Implements Gibbs sampling. (https://en.wikipedia.org/wiki/Gibbs_sampling)
For finite variables, the posterior distribution is sampled by building the exact distribution.
For continuous variables, the posterior distribution is sampled using Metropolis Hastings MCMC.
Discrete variables with infinite support are currently not supported.
The Gibbs Sampler only supports CPDs that return Univariate Distributions. (CPD{D<:UnivariateDistribution})

Sampling requires a GibbsSampler object which contains the parameters for Gibbs sampling.
See the GibbsSampler documentation for parameter details.
"""
function Base.rand(bn::BayesNet, sampler::GibbsSampler, nsamples::Integer)

    return gibbs_sample(bn, nsamples, sampler.burn_in; thinning=sampler.thinning,
		consistent_with=sampler.evidence, variable_order=sampler.variable_order,
		time_limit=sampler.time_limit, error_if_time_out=sampler.error_if_time_out,
		initial_sample=sampler.initial_sample, max_cache_size=sampler.max_cache_size)
end

"""
    NOTE: this is inefficient. Use rand(bn, GibbsSampler, nsamples) whenever you can
"""
function Random.rand!(a::Assignment, bn::BayesNet, sampler::GibbsSampler)
    df = rand(bn, sampler, 1)
    for name in names(bn)
        a[name] = df[1, name]
    end
    a
end
