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

