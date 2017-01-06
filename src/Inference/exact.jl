#
# Exact Inference
#

function exact_inference_old(inf::AbstractInferenceState)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence

    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))
    factors = map(n -> table(bn, n, evidence), nodes)

    # successively remove the hidden nodes
    # order impacts performance, but we currently have no ordering heuristics
    for h in hidden
        # find the facts that contain the hidden variable
        contain_h = filter(f -> h in DataFrames.names(f), factors)
        # remove those factors
        factors = setdiff(factors, contain_h)
        # add the product of those factors to the set
        if !isempty(contain_h)
            push!(factors, sumout(reduce((*), contain_h), h))
        end
    end

    # normalize and remove the leftover variables
    f = reduce((*), factors)
    f = normalize!(by(f, query, df -> DataFrame(p = sum(df[:p]))))
    return f
end

"""
    exact_inference(inf)

Compute P(query | evidence) using factors and variable elimination
"""
function exact_inference(inf::AbstractInferenceState)
    bn = inf.bn
    nodes = names(inf)
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff(nodes, vcat(query, names(evidence)))

    factors = map(n -> Factor(bn, n, evidence), nodes)

    # successively remove the hidden nodes
    # order impacts performance, but we currently have no ordering heuristics
    for h in hidden
        # find the facts that contain the hidden variable
        contain_h = filter(ft -> h in ft, factors)
        # add the product of those factors to the set
        if !isempty(contain_h)
            # remove those factors
            factors = setdiff(factors, contain_h)
            push!(factors, sum(reduce((*), contain_h), h))
        end
    end

    ft = normalize!(reduce((*), factors))
    return ft
end

