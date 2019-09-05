mutable struct ScanGreedyHillClimbing <: GraphSearchStrategy
    cache::ScoreComponentCache
    max_n_parents::Int
    max_depth::Int
    prior::DirichletPrior

    function ScanGreedyHillClimbing(
        cache::ScoreComponentCache;
        max_n_parents::Int=3,
        max_depth::Int=1,
        prior::DirichletPrior=UniformPrior(1.0),
        )

        new(cache, max_n_parents, max_depth, prior)
    end
end

function greedy_score(score_components, n, prior_parent_list, datamat, params::ScanGreedyHillClimbing, ncategories::Vector{Int})
    parent_list = prior_parent_list
    while true

        # Compute score
        best_diff = 0.0
        best_parent_list = parent_list
        for i in 1:n

            # 1) add an edge (j->i)
            if length(parent_list[i]) < params.max_n_parents
                for j in deleteat!(collect(1:n), parent_list[i])
                    if adding_edge_preserves_acyclicity(parent_list, j, i)
                        new_parents = sort!(push!(copy(parent_list[i]), j))
                        #println("A")
                        new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
                        #println("B")
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
                if length(parent_list[j]) < params.max_n_parents
                    deleteat!(parent_list[i], idx)
                    if adding_edge_preserves_acyclicity(parent_list, i, j)
                        sort!(push!(parent_list[j], i))
                        new_diff = bayesian_score_component(i, copy(parent_list[i]), ncategories, datamat, params.prior, params.cache) - score_components[i]
                        new_diff += bayesian_score_component(j, copy(parent_list[j]), ncategories, datamat, params.prior, params.cache) - score_components[j]
                        if new_diff > best_diff
                            best_diff = new_diff
                            best_parent_list = deepcopy(parent_list)
                        end
                        deleteat!(parent_list[j], findall((in)(i), parent_list[j]))
                    end
                    sort!(push!(parent_list[i], j))
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
    sum(score_components), parent_list
end

function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame, params::ScanGreedyHillClimbing;
    ncategories::Vector{Int} = map!(i->infer_number_of_instantiations(data[!,i]), Array{Int}(undef, ncol(data)), 1:ncol(data)),
    )

    n = ncol(data)
    parent_list = map!(i->Int[], Array{Vector{Int}}(undef, n), 1:n)
    datamat = convert(Matrix{Int}, data)'
    score_components = bayesian_score_components(parent_list, ncategories, datamat, params.prior, params.cache)

    # 0 depth
    depth = 0
    best, out_score = greedy_score(score_components, n, parent_list, datamat, params, ncategories)

    # > 1 depth
    greedy_parents = out_score
    while depth < params.max_depth
        depth += 1

        # Scan parameters
        best_parent_list = parent_list
        complete = true
        for i in 1:n

            # 1) add an edge (j->i)
            for j in deleteat!(collect(1:n), parent_list[i])
                if adding_edge_preserves_acyclicity(parent_list, j, i)
                    if length(parent_list[i]) < params.max_n_parents
                        new_parent_list = copy(parent_list)
                        new_parents = sort!(push!(new_parent_list[i], j))
                        new_score, out_score = greedy_score(score_components, n, new_parent_list, datamat, params, ncategories)
                        if new_score > best
                            best = new_score
                            complete = false
                            best_parent_list = new_parent_list
                            greedy_parents = out_score
                        end
                    end
                end
            end

            # 2) did this improve our greedy score?
            # Add the best edge and continue
            parent_list = best_parent_list
        end
        if complete
            break
        end
    end

    # compute the greedy solution
    parent_list = greedy_parents

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
