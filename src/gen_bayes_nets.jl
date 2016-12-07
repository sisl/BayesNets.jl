# Built in (?) BayesNets

"""
Generates a random DiscreteBayesNet

Creates DiscreteBayesNet with num_nodes nodes, each with random values (up to)
max_num_parents and max_num_parents for the number of parents and states
(respectively)
"""
function rand_discrete_bn(num_nodes::Int=16,
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
        # i think its impossible for a cycle to appear based on the
        #  algorithm i've outlines, but . . .
        while true
            n_par = min(length(bn), rand(0:max_num_parents))
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
Given a bn, generate valid query and evidence
"""
function bn_inference_init(bn::BayesNet, num_query::Int=2, num_evidence::Int=3)
    @assert (num_query + num_evidence) <= length(names(bn))

    # yay for convoluted
    non_hidden = sample(names(bn), num_query + num_evidence; replace=false)
    query = non_hidden[1:num_query]
    evidence_nodes = non_hidden[(num_query+1):end]
    evidence = Assignment()

    for ev in evidence_nodes
        ncat = ncategories(get(bn, ev).distributions[1])
        evidence[ev] = sample(1:ncat)
    end

    return (query, evidence)
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

