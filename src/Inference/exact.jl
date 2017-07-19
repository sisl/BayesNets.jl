"""
Exact inference using factors and variable eliminations
"""
struct ExactInference <: InferenceMethod
end
function infer{BN<:DiscreteBayesNet}(im::ExactInference, inf::InferenceState{BN})

    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff(nodes, vcat(query, names(evidence)))

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
infer{BN<:DiscreteBayesNet}(inf::InferenceState{BN}) = infer(ExactInference(), inf)
infer{BN<:DiscreteBayesNet}(bn::BN, query::NodeNameUnion; evidence::Assignment=Assignment()) = infer(ExactInference(), InferenceState(bn, query, evidence))

