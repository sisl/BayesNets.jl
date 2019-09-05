
type GreedyThickThinning <: GraphSearchStrategy
    cache::ScoreComponentCache
    max_n_parents::Int
    prior::DirichletPrior

    function GreedyThickThinning(
        cache::ScoreComponentCache;
        max_n_parents::Int=3,
        prior::DirichletPrior=UniformPrior(1.0),
        )

        new(cache, max_n_parents, prior)
    end
end

function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame, params::GreedyThickThinning;
    ncategories::Vector{Int} = map!(i->infer_number_of_instantiations(data[!,i]), Array{Int}(ncol(data)), 1:ncol(data)),
    )

    n = ncol(data)
    parent_list = map!(i->Int[], Array{Vector{Int}}(n), 1:n)
    datamat = convert(Matrix{Int}, data)'
    score_components = bayesian_score_components(parent_list, ncategories, datamat, params.prior, params.cache)

    while true
        best_diff = 0.0
        best_parent_list = parent_list
        for i in 1:n

            # 1) add an edge (j->i)
            if length(parent_list[i]) < params.max_n_parents
                for j in deleteat!(collect(1:n), parent_list[i])
                    if adding_edge_preserves_acyclicity(parent_list, j, i)
                        new_parents = sort!(push!(copy(parent_list[i]), j))
                        new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
                        if new_component_score - score_components[i] > best_diff
                            best_diff = new_component_score - score_components[i]
                            best_parent_list = deepcopy(parent_list)
                            best_parent_list[i] = new_parents
                        end
                    end
                end
            end
        end

        for i in 1:n
            for (idx, j) in enumerate(parent_list[i])

                new_parents = deleteat!(copy(parent_list[i]), idx)
                new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
                if new_component_score - score_components[i] > best_diff
                    best_diff = new_component_score - score_components[i]
                    best_parent_list = deepcopy(parent_list)
                    best_parent_list[i] = new_parents
                end
            end
        end

        if best_diff > 0.0
            parent_list = best_parent_list
            score_components = bayesian_score_components(parent_list, ncategories, datamat, params.prior, params.cache)
        else
            break
        end
    end

    # construct the BayesNet
    cpds = Array{DiscreteCPD}(n)
    varnames = names(data)
    for i in 1:n
        name = varnames[i]
        parents = varnames[parent_list[i]]
        cpds[i] = fit(DiscreteCPD, data, name, parents, params.prior, parental_ncategories=ncategories[parent_list[i]], target_ncategories=ncategories[i])
    end
    BayesNet(cpds)
end
