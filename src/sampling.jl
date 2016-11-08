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

function sample_posterior_discrete(bn::BayesNet, varname::Symbol, a::Assignment)
   num_categories = # TODO

   markov_blanket_cdps = [get(bn, child_name) for child_name in children(bn, varname)]
   push!(markov_blanket_cdps, get(bn, varname))

   posterior_distribution = zeros(num_categories)
   for index in 1:num_categories
       a[varname] = index
       # Sum logs for numerical stability
       posterior_distribution[index] = exp(sum([logpdf(cdp, a) for cdp in markov_blanket_cdps]))
   end
   posterior_distribution = posterior_distribution / sum(posterior_distribution)

   # Adapted from Distributions.jl, credit to its authors
   p = posterior_distribution
   n = length(p)
   i = 1
   c = p[1]
   u = rand()
   while c < u && i < n
       c += p[i += 1]
   end
   return i

end

"""
Temporarily modifies a, but restores it after computations
TODO check if arguments are copied or passed by reference
"""
function sample_posterior(bn::BayesNet, varname::Symbol, a::Assignment)
    original_value = a[varname]
    new_value = # TODO
    a[varname] = original_value
    return new_value
end

function gibbs_sample_main_loop(bn::BayesNet, nsamples::Integer, sample_skip::Integer, 
start_sample::Assignment, consistent_with::Assignment, variable_order::Nullable{Vector{Symbol}},
time_limit::Nullable{Integer})

    t = Dict{Symbol, Vector{Any}}()
    for name in names(bn)
        t[name] = Any[]
    end

    w = ones(Float64, nsamples) # TODO is this needed
    a = start_sample
    if isnull(variable_order)
         v_order = names(bn)
    else
         v_order = get(variable_order)
    end

    for sample_iter in 1:nsamples
        if isnull(variable_order)
             v_order = shuffle!(v_order)
        end

        for varname in v_order

            if haskey(consistent_with, varname)
                # TODO what to do here?
                cpd = get(bn, varname)
                a[varname] = consistent_with[varname]
                w[i] *= pdf(cpd, a)
            else
                a[varname] = sample_posterior(bn, varname, a)
            end
            push!(t[varname], a[varname])

        end

    end

    t[:p] = w / sum(w)
    convert(DataFrame, t)
end

"""
TODO description

The algorithm will return the samples it has collected when either nsamples samples have been collected or time_limit milliseconds have passed.  If time_limit is not specified then the algorithm will run until nsamples have been collected.

sample_skip: every (sample_skip + 1)th sample will be used, the other sample_skip samples will be thrown out.  The higher the sample_skip, the less correlated samples will be but the longer the computation time per sample.
variable_order: variable_order determines the order of variables changed when generating a new sample, if variable_order is None, then a random variable order will be used for each sample.
"""
function gibbs_sample(bn::BayesNet, nsamples::Integer, burn_in::Integer; sample_skip::Integer=99,
consistent_with::Assignment=Assignment(), variable_order::Nullable{Vector{Symbol}}=Nullable{Vector{Symbol}}(), 
time_limit::Nullable{Integer}=Nullable{Integer}(), error_if_time_out::Bool=true, 
inital_sample::Nullable{Assignment}=Nullable{Assignment}())
    """
    TODO algorithm
    TODO unit test under test/
    TODO come up with an automatic method for setting the burn_in period, look at literatures.  
              Once this is implemented, move burn_in to the default parameters
    """
    # TODO check parameters for correctness, variable_order should be null or provide an order for all variables
    # check that initial_sample is either null or a full assignment that is consistent with the variable consistent_with
   
    # Burn in 
    burn_in_initial_sample = # TODO use rand_table_weighted, should be consistent with the varibale consistent_with
    burn_in_samples, burn_in_time = gibbs_sample_main_loop(bn, burn_in, 1, burn_in_initial_sample, 
                                         consistent_with, variable_order, time_limit)
    if error_if_time_out && ~isnull(time_limit)
        time_limit - burn_in_time > 0 || error("Time expired during Gibbs sampling")
    end
   
    # Real samples
    main_samples_initial_sample = # TODO 
    samples, samples_time = gibbs_sample_main_loop(bn, nsamples, sample_skip, 
                               main_samples_initial_sample, consistent_with, variable_order, remaining_time)
    combined_time = burn_in_time + samples_time
    if error_if_time_out && ~isnull(time_limit)
        combined_time < get(time_limit) || error("Time expired during Gibbs sampling")
    end

    return samples
end
