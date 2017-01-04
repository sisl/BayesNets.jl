#
# Loopy Belief Propagation
#

"""
Get the lambda-message to itself for an evidence node.
If it isn't an evidence node, this will break
"""
@inline function _evidence_lambda(nn::NodeName, evidence::Assignment, ncat::Int)
    z = zeros(ncat)
    z[evidence[nn]] = 1

    return z
end

"""
    loopy_belief(inf, nsamples=500; tol=1e-8, iters_for_convergence=6)

Loopy belief propogation for a network.

Stops early if change in messages < `tol` for `iters_for_convergence'
iterations.
"""
function loopy_belief(inf, nsamples::Int=500;
        tol::Float64=1e-8, iters_for_convergence::Int=6)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence
    evidence_nodes = collect(keys(evidence))

    ncat_lut = Dict(nn => ncategories(bn, nn) for nn in nodes)
    parents = map(nn => parents(bn, nn), nodes)
    children = map(nn => children(bn, nn), nodes)
    factors = map(nn => Factor(bn, nn), nodes)

    # evidence node messages to their selves
    evidence_lambdas = [nn => _evidence_lambda(nn, evidence, ncat_lut[nn])
            for nn in evidence_nodes]
    # the index of each node in evidence (lambda) or zero otherwise
    evidence_index = indexin(nodes, evidence_nodes)

    # the messages being passed from node to node (parents or children)
    # each node has a vector containing the messages from its children
    #  each message is about it, so all have the same length
    lambdas = Dict{(NodeName, NodeName), Vector{Int}}()
    # each node has a vector containing the messages from its parents
    #  each message is about the parent, so they may have different lengths
    pis = Dict{(NodeName, NodeName), Vector{Int}}()

    # init first messages
    # instead of calculating the first pi^t and lambda^t messages per node
    #  set all of them to 1's
    # this gets rid of the edge condition for accumulating pi and lambda for 
    #  first iteration
    # I am most unsure about this...
    for (i, cpd) in enumerate(bn.cpds)
        nn = name(cpd)
        nn_ncat = ncat[nn]
        nn_parents = parents[i]
        nn_children = children[i]

        for ch in nn_children
            # lambda from child to parents (nn)
            lambdas[(ch, nn)] = fill(1/nn_ncat, nn_cat)

            # pi from parent (nn) to children
            #  if evidence node, set to evidence_lambda to avoid recomputing
            ev_i = evidence_index[i]
            pis[(nn, ch)] = (ev_i == 0) ? fill(1/nn_ncat, nn_ncat) :
                evidence_lambdas[ev_i]
        end
    end

    # begin iterating
    # use a circular buffer to keep track of the maximum change
    # in the messages per iteration
    change_index = 1
    change_per_iter = fill(Inf, iters_for_convergence)

    for iter in 1:N
        max_change = -Inf

        for (i, cpd) in enumerate(bn.cpds)
            nn = name(cpd)
            ft = factors[i]
            nn_parents = parents[i]
            nn_children = children[i]

            # build current lambda and pi at time t (from messages @ time t-1)

            # lambda is the product of what all its children are telling it
            #  about each instance of itself
            # nn's index in the evidence
            ev_i = evidence_index[i]
            if ev_i == 0 
                # if it is evidence, then just one value will be non-zero
                # normalization will kick in and do magic ...
                lambda = evidence_lambdas[ev_i]
            else
                lambda = reduce(.*, (lambdas[(ch, nn)] for ch in nn_children))
            end

            # pi is the cpd of the node weighted by the pi message that each
            #  parent sent about itself
            if isempty(nn_parents)
                pi = ft.potential
            else
                # multiply each probability with the pi messages from parents
                #  for that particular instantiation of the parents
                ft_temp = broadcast!(ft_temp, nn_parents,
                        [pis[(pa, nn)] for pa in nn_parents])
                pi = sum!(ft_temp, nn_parents).potential
            end

            # build lambda messages to parents
            for pa in enumerate(nn_parents)
                # same as above:
                # multiply each p(x|parents) by pi of all parents but one
                other_pa = setdiff(nn_parents, [pa])
                lx = broadcast(ft, other_pa, [pis[(p, nn)] for p in other_pa])
                # sum out the other parents
                sum!(lx, other_pa)
                # weight by its current evidence lambda
                broadcast!(lx, nn, lambda)
                # sum out the current node
                sum!(lx, nn)
                normalize!(lx)

                # change in the message
                delta = norm(lx.potential - lambdas(nn, pa))
                if delta > max_change
                    max_change = delta
                end

                lambdas(nn, pa) = lx.potential
            end

            # build pi messages to children
            if ev_i != 0
                # if an evidence node, pi will stay [0 ... 0 1 0... 0 ]
                for ch in nn_children
                    other_ch = setdiff(nn_children, [ch])
                    px = pi .* reduce(.*, (lambdas[(c, nn)] for c in other_ch))
                    normalize!(px, 1)

                    # change in the message
                    delta = norm(px.potential - pis[(nn, ch)], Inf)
                    if delta > max_change
                        max_change = delta
                    end

                    new_pis[(nn, ch)] = p_xc
                end
            end
        end

        change_per_iter[i] = max_change
        # a % b is in [0, b), so add 1
        i = i % iters_for_convergence + 1

        if maximum(change_per_iter) <= tol
            break
        end
    end

    # compute belief P(x|e)
    # lambda and pi one last time
    ft = Factor(bn, query)

    lambda = reduce(.*, (lambdas[(ch, nn)] for ch in nn_children))
    if isempty(nn_parents)
        pi = ft.potential
    else
        ft_temp = broadcast!(ft_temp, nn_parents,
                [pis[(pa, nn)] for pa in nn_parents])
        pi = sum!(ft_temp, nn_parents).potential
    end

    ft = Factor([query], ncat_lut[query])
    ft.potential = lambda .* pi
    normalize!(ft)

    return ft
end

