"""
    _init_gibbs_sample(bn, evidence)

A random sample of non-evidence nodes uniformly over their domain
"""
@inline function _init_gibbs_sample(bn::BayesNet, evidence::Assignment=Assignment())
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
get_mb_cpds(bn::BayesNet, node::Symbol) = [get(bn, n) for n in vcat(children(bn, node), node)]

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
    return Weights(p)
end

"""
    infer(GibbsSampling, state::Assignment, InferenceState)

Run Gibbs sampling for `N` iterations. Each iteration changes one node.

Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
@with_kw mutable struct GibbsSamplingNodewise <: InferenceMethod
    nsamples::Int=2E3
    burn_in::Int=500
    thin::Int=3
    state::Assignment = Assignment()
end
function infer(im::GibbsSamplingNodewise, inf::InferenceState{BN}) where {BN<:DiscreteBayesNet}

    nsamples, burn_in, thin, x = im.nsamples, im.burn_in, im.thin, im.state

    burn_in < nsamples || throw(ArgumentError("Burn in ($burn_in) " *
                "must be less than number of samples ($nsamples)"))

    # if the math doesn't work out correctly, loop a couple more times ...
    total_num_samples = Int(ceil((nsamples-burn_in) / thin))

    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence
    non_evidence = setdiff(nodes, keys(evidence))

    # the current state
    if isempty(im.state)
        im.state = _init_gibbs_sample(bn, inf.evidence)
    else
        _ensure_query_nodes_in_bn_and_not_in_evidence(query, names(bn), evidence)
    end
    x = im.state

    n_cats = Dict(n => ncategories(bn, n) for n in non_evidence)
    # if each non-evidence node is a query variable, and its order as a query
    q_loc = indexin(non_evidence, query)

    ϕ = Factor(query, [n_cats[q] for q in query])
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
            # Shushman : See my comments in loopy_belief.jl
            qi != 0 && qi != nothing && (q_ind[qi] = x[n])

            num_iters += 1

            # kick in the afterburners
            if !after_burn && num_iters > burn_in
                after_burn = true
            end

            # start collecting after the burn in and on the `thin`-th iteration
            if after_burn && ( ((num_iters - burn_in) % thin) == 0 )
                num_smpls += 1

                # sample
                @inbounds ϕ.potential[q_ind...] += 1
            end

            # doubly nested loops!! yay!!
            if num_smpls >= total_num_samples
                finished = true
                break
            end
        end
    end

    normalize!(ϕ)
    return ϕ
end

"""
    infer(im, inf)

Run Gibbs sampling for `N` iterations. Each iteration changes all
nodes.
Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
@with_kw mutable struct GibbsSamplingFull <: InferenceMethod
    nsamples::Int=2E3
    burn_in::Int=500
    thin::Int=3
    state::Assignment=Assignment()
end
function infer(im::GibbsSamplingFull, inf::InferenceState{BN}) where {BN<:DiscreteBayesNet}

    nsamples, burn_in, thin = im.nsamples, im.burn_in, im.thin

    burn_in < nsamples || throw(ArgumentError("Burn in ($burn_in) " *
                "must be less than number of samples ($nsamples)"))

    # if the math doesn't work out correctly, loop a couple more times ...
    total_num_samples = Int(ceil((nsamples-burn_in) / thin))

    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence
    non_evidence = setdiff(nodes, keys(evidence))

    n_cats = Dict(n => ncategories(bn, n) for n in non_evidence)
    # if each non-evidence node is a query variable, and its order as a query
    q_loc = indexin(non_evidence, query)

    # the current state
    if isempty(im.state)
        im.state = _init_gibbs_sample(bn, inf.evidence)
    else
        _ensure_query_nodes_in_bn_and_not_in_evidence(query, names(bn), evidence)
    end
    x = im.state

    ϕ = Factor(query, [n_cats[q] for q in query])
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
            qi != 0 && qi != nothing && (q_ind[qi] = x[n])
        end

        num_iters += 1

        # kick in the afterburners
        if !after_burn && num_iters > burn_in
            after_burn = true
        end

        # start collecting after the burn in and on the `thin`-th iteration
        if after_burn && ( ((num_iters - burn_in) % thin) == 0 )
            num_smpls += 1
            @inbounds ϕ.potential[q_ind...] += 1
        end

        if num_smpls >= total_num_samples
            finished = true
        end
    end

    normalize!(ϕ)
    return ϕ
end

