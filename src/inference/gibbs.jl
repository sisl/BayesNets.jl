#
# Gibbs Sampling code
#
# Gibbs sampling, inference state, and helper functions

"""
    GibbsInferenceState(bn, query, evidence=Assignment())
    GibbsInferenceState(bn, query, evidence, state)

Holds the state for successive Gibbs Sampling
"""
immutable GibbsInferenceState <: AbstractInferenceState
    bn::DiscreteBayesNet
    query::Vector{NodeName}
    evidence::Assignment
    state::Assignment

    function GibbsInferenceState(bn::DiscreteBayesNet, query::NodeNames,
            evidence::Assignment=Assignment())
        state = _init_gibbs_sample(bn, evidence)

        return GibbsInferenceState(bn, query, evidence, state)
    end

    function GibbsInferenceState(bn::DiscreteBayesNet, query::NodeNames,
            evidence::Assignment, state::Assignment)
        if isa(query, NodeName)
            query = [query]
        else
            query = unique(query)
        end

        # check if any queries aren't in the network
        inds = indexin(query, names(bn))
        zero_loc = findnext(inds, 0, 1)
        if zero_loc != 0
            throw(ArgumentError("Query $(query[zero_loc]) is not "
                        * "in the bayes net"))
        end

        # check if any queries are also evidence
        inds = indexin(query, collect(keys(evidence)))
        nonzero_loc = findfirst(inds .> 0)
        if nonzero_loc != 0
            throw(ArgumentError("Query $(query[nonzero_loc]) is part "
                        * "of the evidence"))
        end

        return new(bn, query, evidence, state)
    end
end

Base.convert{I<:AbstractInferenceState}(::Type{I}, inf::GibbsInferenceState) =
    I(inf.bn, inf.query, inf.evidence)

Base.convert(::Type{GibbsInferenceState}, inf::AbstractInferenceState) =
    GibbsInferenceState(inf.bn, inf.query, inf.evidence)

function Base.show(io::IO, inf::GibbsInferenceState)
    println(io, "Query: $(inf.query)")
    println(io, "Evidence:")
    for (k, v) in inf.evidence
        println(io, "  $k => $v")
    end
    println(io, "State:")
    for (k, v) in inf.state
        println(io, "  $k => $v")
    end
end

"""
    _init_gibbs_sample(bn, evidence)

A random sample of non-evidence nodes uniformly over their domain
"""
@inline function _init_gibbs_sample(bn::BayesNet,
        evidence::Assignment=Assignment())
    sample = Assignment()

    for cpd in bn.cpds
        nn = name(cpd)
        if haskey(evidence, nn)
            sample[nn] = evidence[nn]
        else
            # pick a random variable from that distribution
            sample[nn] = rand(1:ncategories(bn, nn))
        end
    end

    return sample
end

"""Get the cpd's of a node and its children"""
get_mb_cpds(bn::BayesNet, node::Symbol) =
        [get(bn, n) for n in vcat(children(bn, node), node)]

"""
    eval_mb_cpd(node, ncategories, assignment, mb_cpds)

Return the potential of all instances of a node given its markove blanket
as a WeightVec:
    P(node | pa_node) * Prod (c in children) P(c | pa_c)

Trys out all possible values of node (assumes categorical)
Assignment should have values for all in the Markov blanket, including the
variable itself.
"""
@inline function eval_mb_cpd(node::Symbol, ncategories::Int,
        assignment::Assignment, mb_cpds::Vector)
    # pdf accepts Assignment only, so swap out current for new value
    old_value = assignment[node]
    p = zeros(ncategories)

    for x in 1:ncategories
        assignment[node] = x

        p[x] = foldl((*), (pdf(cpd, assignment) for cpd in mb_cpds))
    end

    assignment[node] = old_value
    return WeightVec(p)
end

"""
    gibbs_sampling(inf, nsamples=2000; burnin=500, thin=3)

Run Gibbs sampling for `N` iterations. Each iteration changes one node.

Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
function gibbs_sampling(inf::GibbsInferenceState, nsamples::Int=2E3;
        burn_in::Int=500, thin::Int=3)
    burn_in < nsamples || throw(ArgumentError("Burn in ($burn_in) " *
                "must be less than number of samples ($nsamples)"))

    # if the math doesn't work out correctly, loop a couple more times ...
    total_num_samples = Int(ceil((nsamples-burn_in) / thin))

    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence
    non_evidence = setdiff(nodes, keys(evidence))

    n_cats = Dict(n => ncategories(bn, n) for n in non_evidence)
    # if each non-evidence node is a query variable, and its order as a query
    q_loc = indexin(non_evidence, query)

    # the current state
    x = inf.state

    ft = Factor(query, [n_cats[q] for q in query])
    # manual index into factor.potential
    q_ind = [x[q] for q in query]

    # Markov blankets of each node to sample from
    mb_cpds = Dict(n => get_mb_cpds(bn, n) for n = non_evidence)

    finished = false
    after_burn = false
    num_smpls = 0
    num_iters = 0

    # a sample is samples after sampling just one node
    while !finished
        for (i, n) in enumerate(non_evidence)
            # potential of each instance of n
            wv = eval_mb_cpd(n, n_cats[n], x, mb_cpds[n])
            # sample x_n ~ P(X_n|mb(X))
            x[n] = sample(wv)

            # changing one variable at a time, so can just update that node
            qi = q_loc[i]
            # if it is a query variable, update the index
            qi != 0 && (q_ind[qi] = x[n])

            num_iters += 1

            # kick in the afterburners
            if !after_burn && num_iters > burn_in
                after_burn = true
            end

            # start collecting after the burn in and on the `thin`-th iteration
            if after_burn && ( ((num_iters - burn_in) % thin) == 0 )
                num_smpls += 1

                # sample
                @inbounds ft.potential[q_ind...] += 1
            end

            # doubly nested loops!! yay!!
            if num_smpls >= total_num_samples
                finished = true
                break
            end
        end
    end

    normalize!(ft)
    return ft
end

"""
    gibbs_sampling_full(inf, nsamples=2000; burnin=500, thin=3)

Run Gibbs sampling for `N` iterations. Each iteration changes all nodes.
Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
function gibbs_sampling_full(inf::GibbsInferenceState, nsamples::Int=2E3;
        burn_in::Int=500, thin::Int=3)
    burn_in < nsamples || throw(ArgumentError("Burn in ($burn_in) " *
                "must be less than number of samples ($nsamples)"))

    # if the math doesn't work out correctly, loop a couple more times ...
    total_num_samples = Int(ceil((nsamples-burn_in) / thin))

    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence
    non_evidence = setdiff(nodes, keys(evidence))

    n_cats = Dict(n => ncategories(bn, n) for n in non_evidence)
    # if each non-evidence node is a query variable, and its order as a query
    q_loc = indexin(non_evidence, query)

    # the current state
    x = inf.state

    ft = Factor(query, [n_cats[q] for q in query])
    # manual index into factor.potential
    q_ind = [x[q] for q in query]

    # Markov blankets of each node to sample from
    mb_cpds = Dict(n => get_mb_cpds(bn, n) for n = non_evidence)

    finished = false
    after_burn = false
    num_smpls = 0
    num_iters = 0

    while !finished
        # generate a new sample
        for (i, n) in enumerate(non_evidence)
            # potential of each instance of n
            wv = eval_mb_cpd(n, n_cats[n], x, mb_cpds[n])
            # sample x_n ~ P(X_n|mb(X))
            x[n] = sample(wv)

            # check if `n` is a query node, and update the index
            qi = q_loc[i]
            qi != 0 && (q_ind[qi] = x[n])
        end

        num_iters += 1

        # kick in the afterburners
        if !after_burn && num_iters > burn_in
            after_burn = true
        end

        # start collecting after the burn in and on the `thin`-th iteration
        if after_burn && ( ((num_iters - burn_in) % thin) == 0 )
            num_smpls += 1
            @inbounds ft.potential[q_ind...] += 1
        end

        if num_smpls >= total_num_samples
            finished = true
        end
    end

    normalize!(ft)
    return ft
end

