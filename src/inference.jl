function Distributions.ncategories(bn::DiscreteBayesNet, node::Symbol)
    return Distributions.ncategories(get(bn, node).distributions[1])
end

###############################################################################
#                       EXACT INFERENCE
###############################################################################
#Bernoulli does not count as a categoical RV, so that can' tbe added to a BayesNet
"""
Exact inference using factors
"""
function exact_inference(bn::BayesNet, query::Vector{Symbol};
         evidence::Assignment=Assignment())
    # P(query | evidence)

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

    # normalize and remove the leftover variables (I'm looking at you sumout)
    f = foldl((*), factors)
    f = normalize(by(f, query, df -> DataFrame(p = sum(df[:p]))))
    return f
end

function exact_inference(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment())
    return exact_inference(bn, [query]; evidence=evidence)
end

###############################################################################
#                       LIKELIHOOD WEIGHTING
###############################################################################
function weighted_built_in(bn::BayesNet, query::Union{Vector{Symbol}, Symbol};
        evidence::Assignment=Assignment(), N::Int=100)
    samples = rand_table_weighted(bn; consistent_with=evidence, nsamples=N)
    return by(samples, query, df -> DataFrame(probability = sum(df[:p])))
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

"""
Samples has `a`, replaces its value f(samples[a, :probability], v), else adds it

Samples must have a column called probabiliteis

All columns of samples must be in a, but not all columns of a must be
in samples

`f` should be able to take a DataFrames.DataArray as its first element
"""
function update_samples(samples::DataFrame, a::Assignment, v=1, f::Function=+)
    # copied this from filter in factors.jl
    # assume a has a variable for all columns except :probability
    mask = trues(nrow(samples))
    col_names = setdiff(names(samples), [:probability])

    for s in col_names
        mask &= (samples[s] .== a[s])
    end

    # hopefully there is only 1, but this still works else
    if any(mask)
        samples[mask, :probability] = f(samples[mask, :probability], v)
    else
        # get the assignment in the correct order for the dataframe
        new_row = [a[s] for s in col_names]
        push!(samples, @data(vcat(new_row, v)))
    end
end

# increase size of samples as we go . . .
function likelihood_weighting_grow(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N::Int=100)
    nodes = names(bn)
    # hidden nodes in the network
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    # all the samples seen
    samples = DataFrame(push!(fill(Int, length(query)), Float64),
            vcat(query, [:probability]), 0)
    sample = Assignment()

    for i = 1:N
        wt = 1
        # will be in topological order because of
        #  _enforce_topological_order
        for cpd in bn.cpds
            nn = name(cpd)
            if haskey(evidence, nn)
                sample[nn] = evidence[nn]
                # update the weight with the pdf of the conditional
                # prob dist of a node given the currently sampled
                # values and the observed value for that node
                wt *= pdf(cpd, sample)
            else
                sample[nn] = rand(cpd, sample)
            end
        end

        # marginalize on the go
        # samples is over the query variables, sample is not, but it works
        update_samples(samples, sample, wt)
    end

    samples[:probability] /= sum(samples[:probability])
    return samples
end

function likelihood_weighting(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=100)
    return likelihood_weighting(bn, [query]; evidence=evidence, N=N)
end

function likelihood_weighting_grow(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N::Int=100)
    return likelihood_weighting_grow(bn, [query]; evidence=evidence, N=N)
end

###############################################################################
#                       GIBBS SAMPLING
###############################################################################
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
get_mb_cpds(bn, node) = [get(bn, n) for n in push!(children(bn, node), node)]

"""P(node | pa_node) * Prod (c in children) P(c | pa_c)

Trys out all possible values of node (assumes categorical)
Assignment should have values for all nodes in the network
"""
function eval_mb_cpd(node::Symbol, ncategories::Int,
        assignment::Assignment,
        mb_cpds::Vector)
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
Discareds first `burn_in` samples and keeps only the
`thin`-th sample. Ex, if `thin=3`, will discard the first two samples and keep
the third.
"""
function gibbs_sampling(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N=2E3, burn_in=500, thin=1)
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
                # doubly nested loops!! yay!!
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

# each new sample is an iteration of all nodes
function gibbs_sampling_full_iter(bn::BayesNet, query::Vector{Symbol};
        evidence::Assignment=Assignment(), N=2E3, burn_in=500, thin=1)
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

function gibbs_sampling(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(),N=1E3, burn_in=500, thin=3)
    gibbs_sampling(bn, [query]; evidence=evidence, N=N, burn_in=burn_in, thin=thin)
end

function gibbs_sampling_full_iter(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(),N=1E3, burn_in=500, thin=3)
    gibbs_sampling_full_iter(bn, [query]; evidence=evidence, N=N, burn_in=burn_in, thin=thin)
end

###############################################################################
#                       LOOPY BELIEF PROPAGATION
###############################################################################
function _gen_evidence_lambda(nn, evidence, ncat)
    if haskey(evidence, nn)
        z = zeros(ncat)
        z[evidence[nn]] = 1
    else
        z = ones(ncat)
    end

    return z
end

"""
Loopy beleif propogation for a network.

Early stopping if change is messages < `tol` for `iters_for_convergence'
iterations.
"""
function loopy_belief(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N=100,
        tol::Float64=1E-8, iters_for_convergence::Int=6)
    bn_names = names(bn)

    ncat_lut = Dict(nn => ncategories(bn, nn) for nn in bn_names)
    children_lut = Dict(nn => children(bn, nn) for nn in bn_names)
    parents_lut = Dict(nn => parents(bn, nn) for nn in bn_names)
    # evidence node messages to their selves
    # if it isn't an evidence node, all ones vector does nothing
    lambda_self = Dict(nn => _gen_evidence_lambda(nn, evidence, ncat_lut[nn])
            for nn in bn_names)

    # the messages being passed from node to node (parents or children)
    # from child to parent (with info about the parent)
    lambdas = Dict{Tuple{Symbol, Symbol}, Vector{Float64}}()
    # from parent to child (with infor about the parent)
    pis = Dict{Tuple{Symbol, Symbol}, Vector{Float64}}()

    # init first messages
    # Assumes the all previous messages are all ones
    # except for orphan nodes; pi = pdf
     # this is my biggest unknown
    for cpd in bn.cpds
        nn = name(cpd)
        t = table(bn, nn)
     
        # current pi and lambda messages
        # pi is just the p(x) since all parents are equally likely
        pi = by(t, nn, df -> DataFrame(p = sum(df[:p])))
        pi = pi[:p]

        # ideally an if statement if its an evidence node
        #  would be more optimal, but this works too
        lambda = lambda_self[nn]

        # lambdas to parents
        nn_parents = parents_lut[nn]
        for pa in nn_parents
            other_pa = setdiff(nn_parents, [pa])
            # we sum out over the other parents
            l_xp = by(t, [pa, nn], df -> DataFrame(p = sum(df[:p])))
            # weight by its currents lambda
            l_xp[:p] = l_xp[:p] .* lambda[l_xp[nn]]
            # sum over itself
            l_xp = by(l_xp, pa, df -> DataFrame(p = sum(df[:p])))
            # normalize and tell its parents what it really thinks
            lambdas[(nn, pa)] = normalize(l_xp[:p], 1)
        end

        # pi messages to children
        for ch in children_lut[nn]
            # assumes all lambda  messages sent up are all ones
            pis[(nn, ch)] = normalize(pi .* lambda, 1)
        end
    end

    # begin iterating
    # use a circular buffer to keep track of the maximum change
    # in the messages per iteration
    i = 1
    change_per_iter = fill(Inf, iters_for_convergence)

    # pass all messages in parallel, so keep a copy of new messages
    new_lambdas = Dict{Tuple{Symbol, Symbol}, Vector{Float64}}()
    new_pis = Dict{Tuple{Symbol, Symbol}, Vector{Float64}}()

    for iter in 1:N
        max_change = -Inf

        for cpd in bn.cpds
            nn = name(cpd)
            t = table(bn, nn)
            # we use these a couple time
            nn_parents = parents_lut[nn]
            nn_children = children_lut[nn]

            # build current lambda and pi at time t (from messages @ time t)

            # lambda is the product of what all its children are telling it
            #  about each instance of itself
            lambda = lambda_self[nn] .* foldl((.*),
                ones(ncat_lut[nn]),
                (lambdas[(ch, nn)] for ch in nn_children))

            if isempty(nn_parents)
                pi = cpd.distributions[1].p
            else
                # save off the original probabilities for future use
                p_old = deepcopy(t[:p])
                # multiply each probability with the pi messages from parents
                #  for that particular instantiation of the parents
                t[:p] = t[:p] .* foldl((.*), (pis[(p, nn)][t[p]]
                        for p in nn_parents))
                # sum out the parents
                pi = by(t, nn, df -> DataFrame(p = sum(df[:p])))
                pi = Vector(pi[:p])
                # restore balance to the force
                t[:p] = p_old

                # with lambda and pi, compute the lambdas to parents
                #  if no parents, would not loop here, but keep it inside
                #  the if-clause for `p_old`
                for pa in nn_parents
                    # same as above:
                    # multiply each p(x|parents) by pi of all parents but one
                    other_pa = setdiff(nn_parents, [pa])

                    t[:p] = t[:p] .* foldl((.*), ones(t[:p]),
                            (pis[(p, nn)][t[p]] for p in other_pa))
                    # sum out the other parents
                    l_xp = by(t, [pa, nn], df -> DataFrame(p = sum(df[:p])))
                    # weight by its currents lambda
                    l_xp[:p] = l_xp[:p] .* lambda[l_xp[nn]]
                    # sum over itself
                    l_xp = by(l_xp, pa, df -> DataFrame(p = sum(df[:p])))
                    l_xp = normalize(l_xp[:p], 1)

                    # restore the original probabilities
                    t[:p] = p_old

                    # change in the message
                    delta = norm(l_xp - lambdas[(nn, pa)], Inf)
                    if delta > max_change
                        max_change = delta
                    end

                    new_lambdas[(nn, pa)] = l_xp
                end
            end

            # build pi messages to children
            for ch in nn_children
                other_ch = setdiff(nn_children, [ch])
                p_xc = pi .* lambda_self[nn] .* foldl((.*),
                        ones(ncat_lut[nn]),
                        (lambdas[(c, nn)] for c in other_ch))
                normalize!(p_xc, 1)

                # change in the message
                delta = norm(p_xc - pis[(nn, ch)], Inf)
                if delta > max_change
                    max_change = delta
                end

                new_pis[(nn, ch)] = p_xc
            end
        end

        pis = deepcopy(new_pis)
        lambdas = deepcopy(new_lambdas)

        change_per_iter[i] = max_change
        # a % b in [0, b), so add 1
        i = i % iters_for_convergence + 1

        if maximum(change_per_iter) <= tol
            print("ended at iter $(iter)")
            break
        end
    end

    # compute belief P(x|e)
    # lambda and pi one last time
    t = table(bn, query)
    lambda = lambda_self[query] .* foldl((.*),
        ones(ncat_lut[query]),
        (lambdas[(ch, query)] for ch in children_lut[query]))

    if isempty(parents_lut[query])
        pi = get(bn, query).distributions[1].p
    else
        t[:p] = t[:p] .* foldl((.*), (pis[(pa, query)][t[pa]]
                for pa in parents_lut[query]))
        # sum out the parents
        pi = by(t, query, df -> DataFrame(p = sum(df[:p])))
        pi = Vector(pi[:p])
    end

    d = DataFrame()
    d[query] = 1:ncat_lut[query]
    d[:probability] = normalize(lambda .* pi, 1)
    return d
end

