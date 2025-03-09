"""
Exact inference using factors and variable eliminations
"""
struct ExactInference <: InferenceMethod
end
function infer(im::ExactInference, inf::InferenceState{BN}) where {BN<:DiscreteBayesNet}

    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence

    # hidden = setdiff(nodes, vcat(query, keys(evidence)))
    hidden = elimination_order(bn, query, evidence)

    factors = map(n -> Factor(bn, n, evidence), nodes)

    # successively remove the hidden nodes
    # order impacts performance, but we currently have no ordering heuristics
    for h in hidden
        # find the facts that contain the hidden variable
        contain_h = filter(ϕ -> h in ϕ, factors)
        # add the product of those factors to the set
        if !isempty(contain_h)
            # remove those factors
            factors = setdiff(factors, contain_h)
            push!(factors, sum(reduce((*), contain_h), h))
        end
    end

    ϕ = normalize!(reduce((*), factors))
    return ϕ
end
infer(inf::InferenceState{BN}) where {BN<:DiscreteBayesNet} = infer(ExactInference(), inf)
infer(bn::BN, query::NodeNameUnion; evidence::Assignment=Assignment()) where {BN<:DiscreteBayesNet} = infer(ExactInference(), InferenceState(bn, query, evidence))

function elimination_order(bn::BayesNet, query::AbstractVector, evidence::AbstractDict)
    order = Symbol[]
    index = Dict{Symbol, Int}()

    for v in names(bn)
        if !haskey(evidence, v)
            push!(order, v)
            index[v] = length(order)
        end
    end

    # construct reduced graph
    matrix = spzeros(Int, length(order), length(order))

    for v in order
        i = index[v]
        push!(rowvals(matrix), i)
        push!(nonzeros(matrix), 1)

        for w in children(bn, v)
            if haskey(index, w)
                j = index[w]
                push!(rowvals(matrix), j)
                push!(nonzeros(matrix), 1)
            end
        end

        matrix.colptr[i + 1] = length(rowvals(matrix)) + 1
    end

    # moralize graph
    matrix = matrix' * matrix

    # make query variables a clique
    n = length(query)
    clique = Vector{Int}(undef, n)

    for j in 1:n
        clique[j] = index[query[j]]
    end

    matrix[clique, clique] .= 1
    # alg = CliqueTrees.MF()  # minimum fill heuristic
    alg = CliqueTrees.MMD() # minimum degree heuristic
    # alg = CliqueTrees.MCS() # maximum cardinality search
    perm, _ = CliqueTrees.permutation(matrix; alg=CliqueTrees.CompositeRotations(clique, alg))
    return order[perm[1:end - n]]
end
