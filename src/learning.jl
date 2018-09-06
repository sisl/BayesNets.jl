function Distributions.fit(::Type{BayesNet}, data::DataFrame, dag::DAG, cpd_types::Vector{DataType})

    length(cpd_types) == nv(dag) || throw(DimensionMismatch("dag and cpd_types must have the same length"))

    cpds = Array{CPD}(undef, length(cpd_types))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        C = cpd_types[i]
        parents = tablenames[inneighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, dag::DAG, ::Type{C}) where {C<:CPD}

    cpds = Array{C}(undef, nv(dag))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        parents = tablenames[inneighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
Distributions.fit(::Type{BayesNet{T}}, data::DataFrame, dag::DAG) where {T<:CPD} = fit(BayesNet, data, dag, T)

function _get_dag(data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}})
    varnames = names(data)
    dag = DAG(length(varnames))
    for (a,b) in edges
        i = findfirst(isequal(a), varnames)
        j = findfirst(isequal(b), varnames)
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
function Distributions.fit(::Type{BayesNet{T}}, data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}}) where {T<:CPD}
    dag = _get_dag(data, edges)
    fit(BayesNet, data, dag, T)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, edges::Tuple{Vararg{Pair{NodeName, NodeName}}}, cpd_types::Vector{DataType})
    dag = _get_dag(data, edges)
    fit(BayesNet, data, dag, cpd_types)
end
function Distributions.fit(::Type{BayesNet{T}}, data::DataFrame, edge::Pair{NodeName, NodeName}) where {T<:CPD}
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, T)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, edge::Pair{NodeName, NodeName}, cpd_types::Vector{DataType})
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, cpd_types)
end
function Distributions.fit(::Type{BayesNet}, data::DataFrame, edge::Pair{NodeName, NodeName}, ::Type{T}) where {T<:CPD}
    dag = _get_dag(data, tuple(edge))
    fit(BayesNet, data, dag, T)
end

############################

"""
    ScoreComponentCache
Used to store scores in a priority queue such that graph search algorithms know
when a particular construction has already been made.
    cache[ⱼ](parentsⱼ, score) for the ith variable with parents parents
"""
const ScoreComponentCache = Vector{PriorityQueue{Vector{Int}, Float64}} # parent indeces -> score

"""
    ScoreComponentCache(data::DataFrame)
Construct an empty ScoreComponentCache the size of ncol(data)
"""
function ScoreComponentCache(data::DataFrame)
    cache = Array{PriorityQueue{Vector{Int}, Float64}}(undef, ncol(data))
    for i in 1 : ncol(data)
        cache[i] = PriorityQueue{Vector{Int}, Float64, Base.Order.ForwardOrdering}(Base.Order.Forward)
    end
    cache
end

############################

"""
    ScoringFunction
An abstract type for which subtypes allow extracting CPD score components,
which are to be maximized:
score_component(::ScoringFunction, cpd::CPD, data::DataFrame)
"""
abstract type ScoringFunction end

"""
    score_component(a::ScoringFunction, cpd::CPD, data::DataFrame)
Extract a Float64 score for a cpd given the data.
One seeks to maximize the score.
"""
score_component(a::ScoringFunction, cpd::CPD, data::DataFrame) = error("score_component not defined for ScoringFunction $a")

"""
    score_component(a::ScoringFunction, cpd::CPD, data::DataFrame, cache::ScoreComponentCache)
As score_component(ScoringFunction, cpd, data), but returns pre-computed values from the cache
if they exist, and populates the cache if they don't
"""
function _get_parent_indeces(parents::NodeNames, data::DataFrame)
    varnames = names(data)
    retval = Array{Int}(undef, length(parents))
    for (i,p) in enumerate(parents)
        retval[i] = something(findfirst(isequal(p), varnames), 0)
    end
    retval
end
function score_component(
    a::ScoringFunction,
    cpd::CPD,
    data::DataFrame,
    cache::ScoreComponentCache,
    )

    pinds = _get_parent_indeces(parents(cpd), data)
    varnames = names(data)
    i = something(findfirst(isequal(name(cpd)), varnames), 0)

    if !haskey(cache[i], pinds)
        cache[i][pinds] = score_component(a, cpd, data)
    end

    cache[i][pinds]
end

"""
    score_components(a::ScoringFunction, cpd::CPD, data::DataFrame)
    score_components(a::ScoringFunction, cpds::Vector{CPD}, data::DataFrame, cache::ScoreComponentCache)
Get a list of score components for all cpds
"""
function score_components(a::ScoringFunction, cpds::Vector{C}, data::DataFrame) where {C<:CPD}
    retval = Array{Float64}(undef, length(cpds))
    for (i,cpd) in enumerate(cpds)
        retval[i] = score_component(a, cpd, data)
    end
    retval
end
function score_components(a::ScoringFunction, cpds::Vector{C}, data::DataFrame, cache::ScoreComponentCache) where {C<:CPD}
    retval = Array{Float64}(undef, length(cpds))
    for (i,cpd) in enumerate(cpds)
        retval[i] = score_component(a, cpd, data, cache)
    end
    retval
end


"""
    NegativeBayesianInformationCriterion
A ScoringFunction for the negative Bayesian information criterion.

    BIC = -2⋅L + k⋅ln(n)

       L - the log likelihood of the data under the cpd
       k - the number of free parameters to be estimated
       n - the sample size
"""
struct NegativeBayesianInformationCriterion <: ScoringFunction
end
function score_component(::NegativeBayesianInformationCriterion, cpd::CPD, data::DataFrame)
    L = logpdf(cpd, data)
    k = nparams(cpd)
    n = nrow(data)

    bic = -2*L + k*log(n)

    -bic
end

##########################

"""
    fit{C<:CPD}(::Type{BayesNet{C}}, ::DataFrame, ::GraphSearchStrategy)
Run the graph search algorithm defined by GraphSearchStrategy
"""
ProbabilisticGraphicalModels.fit(::Type{BayesNet{C}}, data::DataFrame, params::GraphSearchStrategy) where {C<:CPD} = error("fit not defined for GraphSearchStrategy $params")

"""
    K2GraphSearch
A GraphSearchStrategy following the K2 algorithm.
Takes polynomial time to find the optimal structure assuming
a topological variable ordering.
"""
mutable struct K2GraphSearch <: GraphSearchStrategy
    order::NodeNames            # topological ordering of variables
    cpd_types::Vector{DataType} # cpd types, in same order as `order`
    max_n_parents::Int          # maximum number of parents per CPD
    metric::ScoringFunction     # metric we are trying to maximize

    function K2GraphSearch(
        order::NodeNames,
        cpd_types::Vector{DataType};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
        )

        new(order, cpd_types, max_n_parents, metric)
    end
    function K2GraphSearch(
        order::NodeNames,
        cpdtype::Type{C};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
    ) where {C<:CPD}

        cpd_types = fill(C, length(order))
        new(order, cpd_types, max_n_parents, metric)
    end
    function K2GraphSearch(
        order::NodeNames,
        cpdtype::Type{CategoricalCPD};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
        )

        error("Cannot construct K2GraphSearch with only CategoricalCPD. You must specify the child distribution, CategoricalCPD{D}. For example, CategoricalCPD{Categorical}, aka DiscreteCPD.")
    end
    function K2GraphSearch(
        order::NodeNames,
        cpdtype::Type{StaticCPD};
        max_n_parents::Int=3,
        metric::ScoringFunction=NegativeBayesianInformationCriterion(),
        )

        error("Cannot construct K2GraphSearch with only StaticCPD. You must specify the child distribution, StaticCPD{D}. For example, StaticCPD{Normal}.")
    end
end

function Distributions.fit(::Type{BayesNet{C}}, data::DataFrame, params::K2GraphSearch) where {C<:CPD}

    N = length(params.order)
    cpds = Array{C}(undef, N)
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
