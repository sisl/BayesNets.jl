"""
Exact inference using factors and variable eliminations
"""
type ExactInference <: InferenceMethod
end
function infer(::ExactInference, bn::DiscreteBayesNet, query::Vector{NodeName}; evidence::Assignment=Assignment())

    nodes = names(bn)
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
            push!(factors, sumout(foldl((*), contain_h), h))
        end
    end

    # normalize and remove the leftover variables
    f = foldl((*), factors)
    f = normalize(by(f, query, df -> DataFrame(p = sum(df[:p]))))
    return f
end

