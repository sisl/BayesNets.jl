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

function randomly_select_assignment_from_rand_table_weighted(bn::BayesNet, rand_samples::DataFrame)
    p = rand_samples[:, :p]
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end

    return Assignment(Dict(varname => rand_samples[i, varname] for varname in names(bn)))
end

function sample_posterior_finite(bn::BayesNet, varname::Symbol, a::Assignment, support::AbstractArray)
   markov_blanket_cdps = [get(bn, child_name) for child_name in children(bn, varname)]
   push!(markov_blanket_cdps, get(bn, varname))

   posterior_distribution = zeros(num_categories)
   for index, domain_element in enumerate(support)
       a[varname] = domain_element
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
   return support[i]

end

function sample_posterior_continuous(bn::BayesNet, varname::Symbol, a::Assignment; nsamples::Integer=20)
    # TODO likelihood weighted sampling may not be the correct way to do this
    # TODO if likelihood weighted sampling is used, then you must keep sampling until a sample with non-zero probability occurs

    children_cdps = [get(bn, child_name) for child_name in children(bn, varname)]
    var_cpd = get(bn, varname)

    # TODO if this doesn't get replaced with another method (it likely will) then preallocate everything properly
    t = Dict{Symbol, Vector{Any}}()
    t[varname] = Any[]

    w = ones(Float64, nsamples)

    for i in 1:nsamples
        a[varname] = rand(cpd, a)
        for cpd in children_cdps
            w[i] *= pdf(cpd, a)
        end
        push!(t[varname], a[varname])
    end
    t[:p] = w / sum(w)
    convert(DataFrame, t)

    assignment = randomly_select_assignment_from_rand_table_weighted(bn, t)
    return assignment[varname]
end

"""
Temporarily modifies a, but restores it after computations
TODO check if arguments are copied or passed by reference
"""
function sample_posterior(bn::BayesNet, varname::Symbol, a::Assignment)
    original_value = a[varname]

    cpd = get(bn, varname)
    distribution = cpd(a)
    if hasfinitesupport(distribution)
        new_value = sample_posterior_finite(bn, varname, a, support(distribution))
    else
        new_value = sample_posterior_continuous(bn, varname, a)
    end

    a[varname] = original_value
    return new_value
end

function gibbs_sample_main_loop(bn::BayesNet, nsamples::Integer, sample_skip::Integer, 
start_sample::Assignment, consistent_with::Assignment, variable_order::Nullable{Vector{Symbol}},
time_limit::Nullable{Integer})

    start_time = now()

    t = Dict{Symbol, Vector{Any}}()
    for name in names(bn)
        t[name] = Any[]
    end

    # w = ones(Float64, nsamples) # TODO is this needed - no
    a = start_sample
    if isnull(variable_order)
         v_order = names(bn)
    else
         v_order = get(variable_order)
    end

    for sample_iter in 1:nsamples
        if ~ isnull(time_limit) && (now() - start_time) > get(time_limit)
            break
        end

        if isnull(variable_order)
            v_order = shuffle!(v_order)
        end

        # skip over sample_skip samples
        for skip_iter in 1:sample_skip
            for varname in v_order
                if ~ haskey(consistent_with, varname)
                     a[varname] = sample_posterior(bn, varname, a)
                end
            end

            if isnull(variable_order)
                v_order = shuffle!(v_order)
            end
        end

        for varname in v_order

            if ~ haskey(consistent_with, varname)
                a[varname] = sample_posterior(bn, varname, a)
            end
            # else
                # TODO what to do here? - do nothing
                #cpd = get(bn, varname)
                #a[varname] = consistent_with[varname]
                #w[i] *= pdf(cpd, a)
            # end
            push!(t[varname], a[varname])

        end

    end

    # t[:p] = w / sum(w)
    return convert(DataFrame, t), Integer(now() - start_time)
end

"""
TODO description

First burn_in samples will be sampled and then discrded.  Next additional samples will be draw, 
and every (sample_skip + 1)th sample will be returned while the rest are discarded.

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
    # for burn_in_initial_sample TODO use rand_table_weighted, should be consistent with the varibale consistent_with
    rand_samples = rand_table_weighted(bn, nsamples=10, consistent_with=consistent_with)
    burn_in_initial_samples = randomly_select_assignment_from_rand_table_weighted(bn, rand_samples)
    burn_in_samples, burn_in_time = gibbs_sample_main_loop(bn, burn_in, 0, burn_in_initial_sample, 
                                         consistent_with, variable_order, time_limit)
    remaining_time = Nullable{Integer}()
    if error_if_time_out && ~isnull(time_limit)
        remaining_time = get(time_limit) - burn_in_time
        remaining_time > 0 || error("Time expired during Gibbs sampling")
    end
   
    # Real samples
    main_samples_initial_sample = Assignment(Dict(varname => burn_in_samples[end, varname] for varname in names(bn))) 
    samples, samples_time = gibbs_sample_main_loop(bn, nsamples, sample_skip, 
                               main_samples_initial_sample, consistent_with, variable_order, remaining_time)
    combined_time = burn_in_time + samples_time
    if error_if_time_out && ~isnull(time_limit)
        combined_time < get(time_limit) || error("Time expired during Gibbs sampling")
    end

    # TODO remove rows you conditioned on for interpretability? - no because others don't? is this different?
    return samples
end
