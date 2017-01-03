#
# Gibbs Sampling code
#

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

    function GibbsInferenceState(bn::DiscreteBayesNet, query::Vector{NodeName},
            evidence::Assignment=Assignment())
        state = _init_gibbs_sample(bn, evidence)

        return new(bn, query, evidence, state)
    end

    function GibbsInferenceState(bn::DiscreteBayesNet, query::NodeName,
            evidence::Assignment=Assignment())

        return GibbsInferenceState(bn, [query], evidence)
    end

    function GibbsInferenceState(bn::DiscreteBayesNet, query::Vector{NodeName},
            evidence::Assignment, state::Assignment)

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

Return the probability of all instances of a node given its markove blanket
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

Gibbs sampling. Runs for `N` iterations.
Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
function gibbs_sampling(inf::GibbsInferenceState, nsamples::Int=2E3;
        burn_in=500, thin=3)
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

    factor = Factor(query, [n_cats[q] for q in query])
    # manual index into factor.probability
    q_ind = [x[q] for q in query]

    # Markov blankets of each node to sample from
    mb_cpds = Dict(n => get_mb_cpds(bn, n) for n = non_evidence)

    finished = false
    after_burn = false
    num_smpls = 1
    num_iters = 1

    # a sample is samples after sampling just one node
    while !finished
        for (i, n) in enumerate(non_evidence)
            # probability of each instance of n
            wv = eval_mb_cpd(n, n_cats[n], x, mb_cpds[n])
            # sample x_n ~ P(X_n|mb(X))
            x[n] = sample(wv)

            # start collecting after the burn in and on the `thin`-th iteration
            if after_burn && ( ((num_iters - burn_in) % thin) == 0 )
                # changing one variable at a time, so can just update that node
                qi = q_loc[i]

                if qi != 0
                    # if it is a query variable
                    q_ind[qi] = x[n]
                    factor.probability[q_ind...] += 1
                end

                num_smpls += 1
            end

            num_iters += 1

            # kick in the afterburners
            if !after_burn && num_iters > burn_in
                after_burn = true
            end

            # doubly nested loops!! yay!!
            if num_smpls > total_num_samples
                finished = true
                break
            end
        end
    end

    return factor
end

"""
Gibbs sampleing, but each new state samples is generated after iterating over
all states in the network, not just one.
"""
function gibbs_sampling_full_iter(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N=2E3, burn_in=500, thin=3)
    assert(burn_in < N)

    nodes = names(bn)
    non_evidence = setdiff(nodes, keys(evidence))

    # the current state
    x = initial_sample(bn, evidence)

    # if the math doesn't work out correctly, loop a couple more times . . .
    num_samples = Int(ceil((N-burn_in) / thin))

    # all the samples seen
    samples = DataFrame(fill(Int, length(query)), query, num_samples)

    # markov blankets of each node to sample from
    mb_cpds = Dict((n => get_mb_cpds(bn, n)) for n = non_evidence)

    order = shuffle(non_evidence)

    after_burn = false
    k = 1
    i = 1
    # assume that each sample is after changing one variable
    while true
        # use a random permutation of non-evidence nodes for ordering
        for n in order
            # for all possible value of X (assumes discrete)
            ncat = ncategories(bn, n)
            p = eval_mb_cpd(n, ncat, x, mb_cpds[n])
            # sample x_n ~ P(X_n|mb(X))
            x[n] = Distributions.sample(WeightVec(p))
        end

        if after_burn && ( ((i - burn_in) % thin) == 0)
            for q in query
                samples[k, q] = x[q]
            end

            k += 1
        end

        i += 1

        if !after_burn && i > burn_in
            after_burn = true
        end

        if k > num_samples
            break
        end
    end

    samples = by(samples, query,
        df -> DataFrame(probability = nrow(df)))
    samples[:probability] /= sum(samples[:probability])
    return samples
end

