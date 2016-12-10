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
iterations. For no stopping, use tol < 0.
"""
function loopy_belief(bn::BayesNet, query::Symbol;
        evidence::Assignment=Assignment(), N=500,
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
        # a % b is in [0, b), so add 1
        i = i % iters_for_convergence + 1

        if maximum(change_per_iter) <= tol
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

