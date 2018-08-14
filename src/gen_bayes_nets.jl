#
# Generate BayesNets
#
# Create (random) Bayesian Networks
#
# Also, Bernoulli does not count as a Categoical RV, so they can't be used in
# a DiscreteBayesNet

"""
    rand_discrete_bn(num_nodes16, max_num_parents=3,
            max_num_states=5, connected=true)

Generate a random DiscreteBayesNet.

Creates DiscreteBayesNet with `num_nodes` nodes, with each node having
a random number of states and parents, up to `max_num_parents` and
`max_num_parents`, respectively.
If `connected`, each node (except the first) will be guaranteed at least one
parent, making the graph connected.
"""
function rand_discrete_bn(num_nodes::Int=16,
        max_num_parents::Int=3,
        max_num_states::Int=5,
        connected::Bool=true)
    num_nodes > 0 || throw(ArgumentError("`num_nodes` must be greater than 0"))
    max_num_parents > 0 || throw(ArgumentError("`max_num_parents` must be " *
                "greater than 0"))
    max_num_states > 1  || throw(ArgumentError("`max_num_states` must be " *
                "greater than 1"))

    min_parents = connected ? 1 : 0

    bn = DiscreteBayesNet();

    for i in range(1, length=num_nodes)
        # the symbol to add, NX, where X is its number
        s = Symbol("N", i)
        n_states = rand(2:max_num_states)

        # keep trying different parents until it isn't cyclic
        # it should be impossible for a cycle to appear based on the
        #  algorithm but ...
        while true
            # how many parents we can possibly add
            n_par = rand(0:min(length(bn), max_num_parents))
            parents = sample(names(bn),
                    min(length(bn), max(min_parents, n_par)); replace=false)

            # add the new random cpd with random parents to the network
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
    rand_bn_inference(bn, num_query=2, num_evidence=3)

Generate a random inference state for a Bayesian Network with an evidence
assignment sample uniformly over the chosen nodes' domain.
"""
function rand_bn_inference(bn::BayesNet, num_query::Int=2, num_evidence::Int=3)
    (num_query + num_evidence) <= length(bn) ||
        throw(ArgumentError("Number of qurey and evidence nodes " *
                    "($(num_query + num_evidence)) is greater than number " *
                    "of nodes ($(length(bn)))"))

    # yay for convoluted
    non_hidden = sample(names(bn), num_query + num_evidence; replace=false)
    query = non_hidden[1:num_query]
    evidence_nodes = non_hidden[(num_query+1):end]
    evidence = Assignment()

    for ev in evidence_nodes
        ncat = ncategories(bn, ev)
        evidence[ev] = sample(1:ncat)
    end

    return InferenceState(bn, query, evidence)
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

    # DiscreteBayes don't support Bernouli, has to be Categorical
    push!(sat_fail, rand_cpd(sat_fail, 2, :B, Symbol[]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :S, Symbol[]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :E, [:B, :S]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :D, [:E]))
    push!(sat_fail, rand_cpd(sat_fail, 2, :C, [:E]))

    return sat_fail
end

"""
An ergodic version of the asia network, with the E variable removed

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

