"""
Exact inference using factors and variable eliminations

Returns P(query | evidence)
"""
function exact_inference(bn::BayesNet, query::Vector{NodeName};
         evidence::Assignment=Assignment())
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
            push!(factors, sumout(reduce((*), contain_h), h))
        end
    end

    # normalize and remove the leftover variables
    f = reduce((*), factors)
    f = normalize(by(f, query, df -> DataFrame(p = sum(df[:p]))))
    return f
end

function exact_inference(bn::BayesNet, query::NodeName;
        evidence::Assignment=Assignment())
    return exact_inference(bn, [query]; evidence=evidence)
end

function exact_inference_inf(inf::AbstractInferenceState)
    nodes = names(inf.bn)
    query = names(inf.factor)
    evidence = inf.evidence
    hidden = setdiff(nodes, vcat(query, collect(keys(evidence))))

    factors = map(n -> Factors.Factor(bn, n, evidence), nodes)

    # successively remove the hidden nodes
    # order impacts performance, but we currently have no ordering heuristics
    for h in hidden
        # find the facts that contain the hidden variable
        contain_h = filter(ft -> h in ft, factors)
        # remove those factors
        factors = setdiff(factors, contain_h)
        # add the product of those factors to the set
        if !isempty(contain_h)
            push!(factors, sum(reduce((*), contain_h), h))
        end
    end

    inf.factor = reduce((*), factors)
    return inf
end

