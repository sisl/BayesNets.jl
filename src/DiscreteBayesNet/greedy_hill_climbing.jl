
struct GreedyHillClimbing <: GraphSearchStrategy
    cache::ScoreComponentCache
    max_n_parents::Int
    prior::DirichletPrior

    function GreedyHillClimbing(
        cache::ScoreComponentCache;
        max_n_parents::Int=3,
        prior::DirichletPrior=UniformPrior(1.0),
        )

        new(cache, max_n_parents, prior)
    end
end
function Distributions.fit(::Type{DiscreteCPD},
    data::DataFrame,
    target::NodeName,
    prior::DirichletPrior;
    ncategories::Int = infer_number_of_instantiations(data[!,target]),
    )

    prior_counts = get(prior, ncategories)
    for v in data[target]
        prior_counts[v] += 1.0
    end

    d = Categorical{Float64,Vector{Float64}}(prior_counts ./ sum(prior_counts))
    CategoricalCPD(target, NodeName[], Int[], Categorical{Float64,Vector{Float64}}[d])
end
function Distributions.fit(::Type{DiscreteCPD},
    data::DataFrame,
    target::NodeName,
    parents::NodeNames,
    prior::DirichletPrior;
    parental_ncategories::Vector{Int} = map!(p->infer_number_of_instantiations(data[!,p]), Array{Int}(length(parents)), parents),
    target_ncategories::Int = infer_number_of_instantiations(data[!,target]),
    )

    # with parents

    if isempty(parents)
        return fit(DiscreteCPD, data, target, prior, ncategories=target_ncategories)
    end

    nparents = length(parents)
    dims = [1:parental_ncategories[i] for i in 1:nparents]
    distributions = Array{Categorical{Float64,Vector{Float64}}}(undef, prod(parental_ncategories))
    for (q, parent_instantiation) in enumerate(Iterators.product(dims...))

        prior_counts = get(prior, target_ncategories)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                prior_counts[data[i, target]] += 1.0
            end
        end
        distributions[q] = Categorical{Float64,Vector{Float64}}(prior_counts ./ sum(prior_counts))
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end
function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame, params::GreedyHillClimbing;
    ncategories::Vector{Int} = map!(i->infer_number_of_instantiations(data[!,i]), Array{Int}(undef, ncol(data)), 1:ncol(data)),
    )

    n = ncol(data)
    parent_list = map!(i->Int[], Array{Vector{Int}}(undef, n), 1:n)
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

            # 2) remove an edge
            for (idx, j) in enumerate(parent_list[i])

                new_parents = deleteat!(copy(parent_list[i]), idx)
                new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
                if new_component_score - score_components[i] > best_diff
                    best_diff = new_component_score - score_components[i]
                    best_parent_list = deepcopy(parent_list)
                    best_parent_list[i] = new_parents
                end

                # 3) flip an edge
                new_parent_list = deepcopy(parent_list) # TODO: make this more efficient
                deleteat!(new_parent_list[i], idx)

                if adding_edge_preserves_acyclicity(new_parent_list, i, j)
                    sort!(push!(new_parent_list[j], i))
                    new_diff = bayesian_score_component(i, new_parent_list[i], ncategories, datamat, params.prior, params.cache) - score_components[i]
                    new_diff += bayesian_score_component(j, new_parent_list[j], ncategories, datamat, params.prior, params.cache) - score_components[j]
                    if new_diff > best_diff
                        best_diff = new_diff
                        best_parent_list = new_parent_list
                    end
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
    cpds = Array{DiscreteCPD}(undef, n)
    varnames = names(data)
    for i in 1:n
        name = varnames[i]
        parents = varnames[parent_list[i]]
        cpds[i] = fit(DiscreteCPD, data, name, parents, params.prior, parental_ncategories=ncategories[parent_list[i]], target_ncategories=ncategories[i])
    end
    BayesNet(cpds)
end