type GibbsInference <: InferenceMethod
end # TODO



""" A random sample of all nodes in network, except for evidence nodes

Not uniform, not sure how to randomly sample over domain of distribution
"""
function initial_sample(bn::BayesNet, evidence::Assignment)
    sample = Assignment()

    for cpd in bn.cpds
        nn = name(cpd)
        if haskey(evidence, nn)
            sample[nn] = evidence[nn]
        else
            # uniform sample across domain
            # pick a random distribution for the node
            d = rand(cpd.distributions)
            # pick a random variable from that distribution
            sample[nn] = Distributions.rand(d)
        end
    end

    return sample
end

"""Get the cpd's of a node and its children"""
get_mb_cpds(bn::BayesNet, node::Symbol) =
        [get(bn, n) for n in vcat(children(bn, node), node)]

"""
The probability of all instances of a node given its markove blanket:
    P(node | pa_node) * Prod (c in children) P(c | pa_c)

Trys out all possible values of node (assumes categorical)
Assignment should have values for all in the Markov blanket, including the
variable itself.
"""
function eval_mb_cpd(node::Symbol, ncategories::Int,
        assignment::Assignment, mb_cpds::Vector)
    # pdf accepts Assignment only, so swap out current for new value
    old_value = assignment[node]
    p = zeros(ncategories)

    for x in range(1, ncategories)
        assignment[node] = x

        p[x] = foldl((*), (pdf(cpd, assignment) for cpd in mb_cpds))
    end

    assignment[node] = old_value
    return p / sum(p)
end

"""
Gibbs sampling. Runs for `N` iterations.
Discareds first `burn_in` samples and keeps only the `thin`-th sample.
Ex, if `thin=3`, will discard the first two samples and keep the third.
"""
function gibbs_sampling(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N=2E3, burn_in=500, thin=3)
    assert(burn_in < N)

    nodes = names(bn)
    non_evidence = setdiff(nodes, keys(evidence))

    # the current state
    x = initial_sample(bn, evidence)

    # if the math doesn't work out correctly, loop a couple more times ...
    num_samples = Int(ceil((N-burn_in) / thin))

    # all the samples seen
    samples = DataFrame(fill(Int, length(query)), query, num_samples)

    # Markov blankets of each node to sample from
    mb_cpds = Dict((n => get_mb_cpds(bn, n)) for n = non_evidence)

    order = shuffle(non_evidence)

    finished = false
    after_burn = false
    k = 1
    i = 1
    # assume that each sample is after changing one variable
    while !finished
        # use a random permutation of non-evidence nodes for ordering
        for n in order
            # for all possible value of X (assumes discrete)
            ncat = ncategories(bn, n)
            p = eval_mb_cpd(n, ncat, x, mb_cpds[n])
            # sample x_n ~ P(X_n|mb(X))
            x[n] = Distributions.sample(WeightVec(p))

            # start collecting after the burn in and on the `thin`-th iteration
            if after_burn && ( ((i - burn_in) % thin) == 0)
                for q in query
                    samples[k, q] = x[q]
                end

                k += 1
            end

            i += 1

            # kick in the afterburners
            if !after_burn && i > burn_in
                after_burn = true
            end

            # doubly nested loops!! yay!!
            if k > num_samples
                finished = true
                break
            end
        end
    end

    samples = by(samples, query,
        df -> DataFrame(probability = nrow(df)))
    samples[:probability] /= sum(samples[:probability])
    return samples
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

# versions to accept just one query variable, instead of a vector
function gibbs_sampling(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(),N=1E3, burn_in=500, thin=3)
    gibbs_sampling(bn, [query]; evidence=evidence, N=N, burn_in=burn_in, thin=thin)
end

function gibbs_sampling_full_iter(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(),N=1E3, burn_in=500, thin=3)
    gibbs_sampling_full_iter(bn, [query]; evidence=evidence, N=N, burn_in=burn_in, thin=thin)
end

