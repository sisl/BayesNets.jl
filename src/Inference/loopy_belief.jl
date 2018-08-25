"""
Loopy belief propogation for a network.

Early stopping if change is messages < `tol` for `iters_for_convergence'
iterations. For no stopping, use tol < 0.
"""
@with_kw struct LoopyBelief <: InferenceMethod
    nsamples::Int = 500
    tol::Float64 = 1e-8
    iters_for_convergence::Int = 6
end

"""
Get the lambda-message to itself for an evidence node.
If it isn't an evidence node, this will break
"""
@inline function _evidence_lambda(nn::NodeName, evidence::Assignment, ncat::Int)
    z = zeros(ncat)
    z[evidence[nn]] = 1

    return z
end

function infer(im::LoopyBelief, inf::InferenceState{BN}) where {BN<:DiscreteBayesNet}

    length(inf.query) == 1 || throw(ArgumentError("There can only be one query variable"))

    nsamples, tol, iters_for_convergence = im.nsamples, im.tol, im.iters_for_convergence

    bn = inf.pgm
    nodes = names(bn)
    query = first(inf.query)
    evidence = inf.evidence
    evidence_nodes = collect(keys(evidence))

    ncat_lut = Dict(nn => ncategories(bn, nn) for nn in nodes)
    parents_lut = map(nn -> parents(bn, nn), nodes)
    children_lut = map(nn -> children(bn, nn), nodes)
    factors = map(nn -> Factor(bn, nn), nodes)

    # evidence node messages to their selves
    evidence_lambdas = map(nn -> _evidence_lambda(nn, evidence, ncat_lut[nn]), evidence_nodes)

    # the index of each node in evidence (lambda) or nothing otherwise
    evidence_index = indexin(nodes, evidence_nodes)

    #=
    The messages being passed from node to node (parents or children)
        Each node has a vector containing the messages from its children
        Each message is about it, so all have the same length
    =#
    lambdas = Dict{Tuple{NodeName, NodeName}, Vector{Float64}}()

    #=
    Each node has a vector containing the messages from its parents
    Each message is about the parent, so they may have different lengths
    =#
    pis = empty(lambdas)

    #=
    Init first messages
        Instead of calculating the first πᵗ and λᵗ messages per node set all of them to 1's
        This gets rid of the edge condition for accumulating π and λ for first iteration
        I am most unsure about this...
    =#
    for (i, cpd) in enumerate(bn.cpds)

        nn = name(cpd)
        nn_ncat = ncat_lut[nn]
        nn_parents = parents_lut[i]
        nn_children = children_lut[i]

        for ch in nn_children
            # lambda from child to parents (nn)
            lambdas[(ch, nn)] = fill(1/nn_ncat, nn_ncat)

            # pi from parent (nn) to children
            #  if evidence node, set to evidence_lambda to avoid recomputing
            ev_i = evidence_index[i]
            pis[(nn, ch)] = isa(ev_i, Int64) ? evidence_lambdas[ev_i] : fill(1/nn_ncat, nn_ncat)
        end
    end

    # messages are passed in parallel, so need a "new" set
    new_lambdas = empty(lambdas)
    # pi messages for evidence nodes won't be (re-)computed, so
    # just copy over *all* initial pi messages
    new_pis = deepcopy(pis)

    # begin iterating
    # use a circular buffer to keep track of the maximum change
    # in the messages per iteration
    change_index = 1
    change_per_iter = fill(Inf, iters_for_convergence)

    for iter in 1:nsamples
        max_change = -Inf

        for (i, cpd) in enumerate(bn.cpds)
            nn = name(cpd)
            nn_ncat = ncat_lut[nn]
            ϕ = factors[i]
            nn_parents = parents_lut[i]
            nn_children = children_lut[i]

            # build current lambda and pi at time t (from messages @ time t-1)

            # lambda is the product of what all its children are telling it
            #  about each instance of itself
            # nn's index in the evidence
            ev_i = evidence_index[i]

            if isa(ev_i, Int64)
                # if it is evidence, then just one value will be non-zero
                # normalization will kick in and do magic ...
                lambda = evidence_lambdas[ev_i]
            else
                lambda = isempty(nn_children) ? ones(nn_ncat) :
                    reduce((x,y) -> broadcast(*, x, y), (lambdas[(ch, nn)] for ch in nn_children))
            end

            # pi is the cpd of the node weighted by the pi message that each
            #  parent sent about itself
            if isempty(nn_parents)
                pi = ϕ.potential
            else
                # multiply each probability with the pi messages from parents
                #  for that particular instantiation of the parents
                ft_temp = broadcast(*, ϕ, nn_parents,
                        [pis[(pa, nn)] for pa in nn_parents])
                pi = sum!(ft_temp, nn_parents).potential
            end

            # build lambda messages to parents
            for pa in nn_parents
                # same as above:
                # multiply each p(x|parents) by pi of all parents but one
                other_pa = setdiff(nn_parents, [pa])
                lx = isempty(other_pa) ? deepcopy(ϕ) :
                        broadcast(*, ϕ, other_pa,
                                [pis[(p, nn)] for p in other_pa])
                # sum out the other parents
                sum!(lx, other_pa)
                # weight by its current evidence lambda
                broadcast!(*, lx, nn, lambda)
                # sum out the current node
                sum!(lx, nn)
                normalize!(lx)

                new_lambdas[(nn, pa)] = lx.potential

                # find change in the message
                delta = norm(lx.potential - lambdas[(nn, pa)])
                if delta > max_change
                    max_change = delta
                end
            end

            # build pi messages to children
            if ev_i == nothing
                # if an evidence node, pi will stay [0 ... 0 1 0... 0 ]
                for ch in nn_children
                    other_ch = setdiff(nn_children, [ch])
                    px = pi
                    # reduce freaks out with empty arrays
                    isempty(other_ch) ||
                        (px .*= reduce((x,y) -> broadcast(*, x, y),
                                    (lambdas[(c, nn)] for c in other_ch)))
                    normalize!(px, 1)

                    new_pis[(nn, ch)] = px

                    # find change in the message
                    delta = norm(px - pis[(nn, ch)], Inf)
                    if delta > max_change
                        max_change = delta
                    end
                end
            end
        end

        # swap 'em
        new_pis, pis = pis, new_pis
        new_lambdas, lambdas, = lambdas, new_lambdas

        change_per_iter[change_index] = max_change
        # a % b is in [0, b), so add 1
        change_index = change_index % iters_for_convergence + 1

        if maximum(change_per_iter) <= tol
            break
        end
    end

    # compute belief P(x|e)
    qi = findfirst(isequal(query), nodes)

    # lambda and pi one last time
    ϕ = factors[qi]
    nn_ncat = ncat_lut[query]
    nn_children = children_lut[qi]
    nn_parents = parents_lut[qi]

    lambda = isempty(nn_children) ? ones(nn_ncat) :
            reduce((x,y) -> broadcast(*, x, y), (lambdas[(ch, query)] for ch in nn_children))

    if isempty(nn_parents)
        pi = ϕ.potential
    else
        broadcast!(*, ϕ, nn_parents, [pis[(pa, query)] for pa in nn_parents])
        pi = sum!(ϕ, nn_parents).potential
    end

    ftr = Factor([query], broadcast(*, lambda, pi))
    normalize!(ftr)

    return ftr
end

