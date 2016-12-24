"""
Exact inference using factors and variable eliminations

Returns P(query | evidence)
"""
function exact_inference(bn::BayesNet, qu::Vector{NodeName};
         ev::Assignment=Assignment())
    nodes = names(bn)
    hidden = setdiff(nodes, vcat(qu, collect(keys(ev))))
    factors = map(n -> table(bn, n, ev), nodes)

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
    f = normalize(by(f, qu, df -> DataFrame(p = sum(df[:p]))))
    return f
end

function exact_inference(bn::BayesNet, qu::NodeName;
        ev::Assignment=Assignment())
    return exact_inference(bn, [qu], ev=ev)
end

function exact_inference_inf(inf::AbstractInferenceState)
    bn = inf.bn
    nodes = names(inf)
    qu = query(inf)
    ev = evidence(inf)
    hidden = setdiff(nodes, vcat(query, names(ev)))

    factors = map(n -> Factors.Factor(bn, n, ev), nodes)

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

