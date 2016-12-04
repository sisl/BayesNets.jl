
function random_discrete_bn(num_nodes::Int=16,
        max_num_parents::Int=3,
        max_num_states::Int=5)
    @assert(num_nodes > 0)
    @assert(max_num_parents > 0)
    @assert(max_num_states > 1)

    bn = DiscreteBayesNet();

    for i in range(1, num_nodes)
        s = Symbol("N", i)
        n_states = rand(2:max_num_states)

        # keep trying different parents until it isn't cyclic
        while true
            n_par = min(length(bn), rand(1:max_num_parents))
            parents = names(bn)[randperm(length(bn))[1:n_par]]

            push!(bn, rand_cpd(bn, n_states, s, parents))

            if is_cyclic(bn.dag)
                delete!(bn, s)
            else
                break
            end
        end
    end

    return bn
end

###############################################################################
#                       EXACT INFERENCE
###############################################################################
"""
"""
# P(query | evidence)
function exact_inference(bn::BayesNet, query::Vector{Symbol};
         evidence::Assignment=Assignment(), ordering=[])

    if !isempty(x)
        throw(DomainError())
    end
    nodes = names(bn)
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    factors = map(n -> table(bn, n, evidence), nodes)

    # order impacts performance, so choose a random ordering :)
#----------------------------------------------------------------------------------
# add in the node ordering stuff
#----------------------------------------------------------------------------------
    for h in hidden
        contain_h = filter(f -> h in DataFrames.names(f), factors)
        factors = setdiff(factors, contain_h)
        if !isempty(contain_h)
            push!(factors, sumout(foldl((*), contain_h), h))
        end
    end

    return normalize(foldl((*), factors))
end

function exact_inference(bn::BayesNet, query::Symbol;
        evidence::Dict{Symbol, Bool}, ordering=[])
    exact_inference(bn, [query], evidence)
end

###############################################################################
#                       LIKELIHOOD WEIGHTING
###############################################################################
function weighted_built_in(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=100)
    samples = bn.rand_table_weighted(bn; consistent_with=evidence, nsamples=N)
    return by(samples, query, df -> DataFrame(p = sum(df[:p])))
end


# structure copied from the rand(bn) and rand_table_weighted function in the
#  BayesNets (sampling.jl)
function weighted_sample(bn::BayesNet, evidence::Assignment=Assignment())
    weight = 1
    sample = Assignment()

    # will be in topological order because of _enforce_topological_order
    # right?
    for cpd in bn.cpds
        nn = name(cpd)
        if nn in keys(evidence)
            sample[nn] = evidence[nn]
            # update the weight with the pdf of the conditional prob dist of a
            #  node given the currently sampled values and the observed value
            #  for that node
            weight *= pdf(cpd, sample[nn])
        else
            sample[nn] = rand(cpd, sample)
        end
    end
    (sample, weight)
end

"""
Approximates p(query|evidence) with N weighted samples using likelihood
weighted sampling
"""
function likelihood_weighting(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=100)
    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))

    # ideally this would be a data frame, but I don't know of an efficient
    # way to check if a row already exists in a datafram
    # dict of (vars => value) => weight
    samples = Dict{Assignment, Float64}()
    for i = 1:N
        smpl, wt = weighted_sample(bn, evidence)
        # perform marginalization manually by adding weights to other samples
        #  with same values of the queries
        q_smpl = Dict(q => smpl[q] for q in query)
        samples[q_smpl] = wt + get(samples, q_smpl, 0)
    end

    # pairs of assignments and their weights 
    pairs = collect(samples)
    # the actual samples
    smpls = map(first, pairs)
    # weight of each assignment
    wts = map(last, pairs)
    # make a data frame
    table = DataFrame(smpls)
    # add the weights and make em probabilities
    table[:p] = wts ./ sum(wts)

    return table
end

function likelihood_weighting(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=100)
    likelihood_weighting(bn, [query], evidence, N)
end

###############################################################################
#                       GIBBS SAMPLING
###############################################################################
"""
"""
function gibbs_sampling(bn::BayesNet, query::Vector{Symbol},
        evidence::Assignment=Assignment(), N=100)

end


###############################################################################
#                       LOOPY BELIEF PROPAGATION
###############################################################################
"""
"""
function loopy_belief_prop(bn::BayesNet, query::Vector{Symbol},
        evidence::Assignment=Assignment(), N=100)

end

