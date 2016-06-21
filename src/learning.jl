function Distributions.fit(::Type{BayesNet}, data::DataFrame, dag::DAG, cpd_types::Vector{DataType})

    length(cpd_types) == nv(dag) || throw(DimensionMismatch("dag and cpd_types must have the same length"))

    cpds = Array(CPD, length(cpd_types))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        C = cpd_types[i]
        parents = tablenames[in_neighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
function Distributions.fit{C<:CPD}(::Type{BayesNet}, data::DataFrame, dag::DAG, ::Type{C})

    cpds = Array(C, nv(dag))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        parents = tablenames[in_neighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
Distributions.fit{T<:CPD}(::Type{BayesNet{T}}, data::DataFrame, dag::DAG) = fit(BayesNet, data, dag, T)

function _get_dag(data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}})
    varnames = names(data)
    dag = DAG(length(varnames))
    for (a,b) in edges
        i = findfirst(varnames, a)
        j = findfirst(varnames, b)
        add_edge!(dag, i, j)
    end
    dag
end

"""
    fit(::Type{BayesNet}, data, edges)
Fit a Bayesian Net whose variables are the columns in data
and whose edges are given in edges

    ex: fit(DiscreteBayesNet, data, (:A=>:B, :C=>B))
"""
function Distributions.fit{T<:CPD}(::Type{BayesNet{T}}, data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}})
    dag = _get_dag(data, edges)
    fit(BayesNet, data, dag, T)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}}, cpd_types::Vector{DataType})
    dag = _get_dag(data, edges)
    fit(BayesNet, data, dag, cpd_types)
end
function Distributions.fit{T<:CPD}(::Type{BayesNet{T}}, data::DataFrame, edge::Pair{NodeName, NodeName})
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, T)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, edge::Pair{NodeName, NodeName}, cpd_types::Vector{DataType})
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, cpd_types)
end
function Distributions.fit{T<:CPD}(::Type{BayesNet}, data::DataFrame, edge::Pair{NodeName, NodeName}, ::Type{T})
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, T)
end

############################

abstract ScoringFunction

type NegativeBayesianInformationCriterion <: ScoringFunction
end

"""
    score_component(::ScoringFunction, cpd::CPD, data::DataFrame)
Return the negative Bayesian information criterion score component

    BIC = -2⋅L + k⋅ln(n)

       L - the log likelihood of the data under the cpd
       k - the number of free parameters to be estimated
       n - the sample size
"""
function score_component(::NegativeBayesianInformationCriterion, cpd::CPD, data::DataFrame)
    L = logpdf(cpd, data)
    k = nparams(cpd)
    n = nrow(data)

    bic = -2*L + k*log(n)

    -bic
end

##########################

abstract GraphSearchStrategy

type K2GraphSearch <: GraphSearchStrategy
    order::Vector{NodeName}     # topological ordering of variables
    cpd_types::Vector{DataType} # cpd types, in same order as `order`
    max_n_parents::Int
    metric::ScoringFunction     # metric we are trying to maximize

    function K2GraphSearch(
        order::Vector{NodeName},
        cpd_types::Vector{DataType};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
        )

        new(order, cpd_types, max_n_parents, metric)
    end
    function K2GraphSearch{C<:CPD}(
        order::Vector{NodeName},
        cpdtype::Type{C};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
        )

        cpd_types = fill(C, length(order))
        new(order, cpd_types, max_n_parents, metric)
    end
end

"""
    Distributions.fit(::Type{BayesNet}, data::DataFrame, params::K2GraphSearch)
Runs the K2 structure search algorithm on the data with the given cpd types and fixed ordering
"""
function Distributions.fit{C<:CPD}(::Type{BayesNet{C}}, data::DataFrame, params::K2GraphSearch)

    N = length(params.order)
    cpds = Array(C, N)
    for i in 1 : N

        cpd_type = params.cpd_types[i]
        target = params.order[i]

        best_score = -Inf

        # find the best parent to add
        potential_parents = params.order[1:i-1]
        for nparents in 0:min(params.max_n_parents,i-1)
            for parents in subsets(potential_parents, nparents)
                cpd = fit(cpd_type, data, target, parents)
                score = score_component(params.metric, cpd, data)

                if score > best_score
                    best_score = score
                    cpds[i] = cpd
                end
            end
        end
    end

    BayesNet(cpds)
end
Distributions.fit(::Type{BayesNet}, data::DataFrame, params::K2GraphSearch) = fit(BayesNet{CPD}, data, params)