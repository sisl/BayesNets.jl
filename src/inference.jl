
###############################################################################
#                       EXACT INFERENCE
###############################################################################
"""
Table only works on DiscreteBayesNets, so . . ..
Also, Bernoulli does not count as a categoical RV, so that can' tbe added to a
DiscreteBayesNet
"""
# P(query | evidence)
function exact_inference(bn::BayesNet, query::Vector{Symbol};
         evidence::Assignment=Assignment(), ordering=[])

#----------------------------------------------------------------------------------
# add in the node ordering stuff here
#----------------------------------------------------------------------------------
    if !isempty(ordering)
        throw(DomainError())
    end

    nodes = names(bn)
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    factors = map(n -> table(bn, n, evidence), nodes)

    # order impacts performance, so choose a random ordering :)
    for h in hidden
        contain_h = filter(f -> h in DataFrames.names(f), factors)
        # remove the factors that contain the hidden variable
        factors = setdiff(factors, contain_h)
        # add the product of the factors to the set
        if !isempty(contain_h)
            push!(factors, sumout(foldl((*), contain_h), h))
        end
    end

    return normalize(foldl((*), factors))
end

function exact_inference(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), ordering=[])
    return exact_inference(bn, [query]; evidence=evidence, ordering=ordering)
end

###############################################################################
#                       LIKELIHOOD WEIGHTING
###############################################################################
function weighted_built_in(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=100)
    samples = rand_table_weighted(bn; consistent_with=evidence, nsamples=N)
    return by(samples, query, df -> DataFrame(p = sum(df[:p])))
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
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:probability]), N)
    samples[:probability] = 1
    sample = Assignment()

    for i = 1:N

        # will be in topological order because of
        #  _enforce_topological_order
        for cpd in bn.cpds
            nn = name(cpd)
            if haskey(evidence, nn)
                sample[nn] = evidence[nn]
                # update the weight with the pdf of the conditional
                # prob dist of a node given the currently sampled
                # values and the observed value for that node
                samples[i, :probability] *= pdf(cpd, sample)
            else
                sample[nn] = rand(cpd, sample)
            end
        end

        # for some reason, you cannot set an entire row in a dataframe
        for q in query
            samples[i, q] = sample[q]
        end
    end

    samples = by(samples, query,
        df -> DataFrame(probability = sum(df[:probability])))
    samples[:probability] /= sum(samples[:probability])
    return samples
end

function likelihood_weighting(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=100)
    return likelihood_weighting(bn, [query]; evidence=evidence, N=N)
end

###############################################################################
#                       GIBBS SAMPLING
###############################################################################
# P(x | markov_blanket(x)) as a factor
function markov_blanket_factor(bn::BayesNet, node::NodeName,
        evidence::Assignment=Assignment())
    cs = children(bn, node)
    t = table(bn, node, evidence)

    if ~isempty(cs)
        t = t * foldl((*), [table(bn, c, evidence) for c in cs])
    end

    return t
end

# not a real random sample, instead chooses random values for
#  non-evidence nodes
function _initial_sample(bn::BayesNet, evidence::Assignment)
    sample = Assignment()

    for cpd in bn.cpds
        nn = name(cpd)
        if haskey(evidence, nn)
            sample[nn] = evidence[nn]
        else
            # random sample from the cpd
            # ideall this would be uniform for each variable, but i don't
            #  know how to access each variable's domain
            sample[nn] = rand(cpd, sample)
        end
    end

    return sample
end

"""
Gibbs sampling. Runs for `N` iterations.
Discareds first `burn_in` samples and keeps only the 
`thin`-th sample. Ex, if `thin=3`, will discard the first two samples and keep
the third.
"""
function gibbs_sampling(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N=1E3, burn_in=500, thin=3)
    assert(burn_in < N)

    nodes = names(bn)
    non_evidence = setdiff(nodes, keys(evidence))

    x = _initial_sample(bn, evidence)

    num_samples = Int(ceil((N-burn_in) / thin))
    # all the samples seen
    samples = DataFrame(fill(Int, length(query)), query, num_samples)

    # markov blankets of each node to sample from (assumes all are discrete)
    mb_factor_lut = Dict{Symbol, DataFrame}()
    for n in non_evidence
        mb_factor_lut[n] = markov_blanket_factor(bn, n, evidence)
    end

    after_burn = false
    k = 0
    for i = 1:N
        # use a random permutation of non-evidence nodes for ordering
        for n in shuffle(non_evidence)
            # x without the current node
            x_prime = Assignment([n => x[n] for n in setdiff(keys(x), [n])])
            p_n = normalize(select(mb_factor_lut[n], x_prime))
            # sample x_n ~ P(X_n|mb(X))
            x[n] = Distributions.sample(p_n[n], WeightVec(p_n[:p].data))
        end

        if !after_burn && i > burn_in
            after_burn = true
        end
        if after_burn && ( ((i-burn_in) % thin) == 0)
            k = Int((i - burn_in) / thin)
            for q in query
                samples[k, q] = x[q]
            end
        end
    end

    samples = by(samples, query,
        df -> DataFrame(probability = nrow(df)))
    samples[:probability] /= sum(samples[:probability])
    return samples
end

function gibbs_sampling(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(),N=1E3, burn_in=500, thin=3)
    gibbs_sampling(bn, [query], evidence; N, burn_in, thin)
end

###############################################################################
#                       LOOPY BELIEF PROPAGATION
###############################################################################
"""
"""
function loopy_belief_prop(bn::BayesNet, query::Vector{Symbol},
        evidence::Assignment=Assignment(), N=100)

end

###############################################################################
#                       GEN BAYES NETS
###############################################################################
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

"""
The usual sprinkler problem
"""
function get_sprinkler_bn()
    sprinkler = DiscreteBayesNet()

    push!(sprinkler, rand_cpd(sprinkler, 2, :Rain, Symbol[]))
    push!(sprinkler, rand_cpd(sprinkler, 2, :Sprinkler, Symbol[:Rain]))
    push!(sprinkler, rand_cpd(sprinkler, 2, :WetGrass,
        Symbol[:Sprinkler, :Rain]))

    return sprinkler
end

"""
Satellite failure network from DMU, pg 17
"""
function get_sat_fail_bn()
    sat_fail = DiscreteBayesNet()

    # DiscreteBayes don't support Benouli, has to be Catagorical
    push!(sat_fail, rand_cpd(sat_fail, 2, :B, Symbol[]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :S, Symbol[]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :E, [:B, :S]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :D, [:E]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :C, [:E]))

    return sat_fail
end

"""
a factored version of the asia network because the boolean nature of the E
variable negatively affects gibbs sampleing

Orignal network: Lauritzen, Steffen L. and David J. Spiegelhalter, 1988
"""
function get_asia_bn()
    asia = DiscreteBayesNet()

    push!(asia, CategoricalCPD(:A, Symbol[], Int64[],
        [Categorical([0.99, 0.01])]))
    push!(asia, CategoricalCPD(:S, Symbol[], Int64[],
        [Categorical([0.5, 0.5])]))
    push!(asia, CategoricalCPD(:T, [:A], [2],
        [Categorical([0.99, 0.01]), Categorical([0.95, 0.05])]))
    push!(asia, CategoricalCPD(:L, [:S], [2],
        [Categorical([0.99, 0.01]), Categorical([0.9, 0.1])]))
    push!(asia, CategoricalCPD(:B, [:S], [2],
        [Categorical([0.97, 0.03]), Categorical([0.4, 0.6])]))

    push!(asia, CategoricalCPD(:X, [:T, :L], [2, 2], [
        Categorical([0.95, 0.05]), # false, false
        Categorical([0.02, 0.98]), # true, false
        Categorical([0.02, 0.98]), # false, true
        Categorical([0.02, 0.98]), # true, true
        ]))

    push!(asia, CategoricalCPD(:D, [:T, :L, :B], [2, 2, 2],[
        Categorical([0.9, 0.1]), # false, false, false
        Categorical([0.2, 0.8]), # true, false, false
        Categorical([0.3, 0.7]), # false, true, false
        Categorical([0.1, 0.9]), # true, true, false

        Categorical([0.3, 0.7]), # false, false, true
        Categorical([0.1, 0.9]), # true, false, true
        Categorical([0.3, 0.7]), # false, true, true
        Categorical([0.1, 0.9]), # true, true, true
        ]))

    return asia
end

