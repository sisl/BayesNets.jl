"""
Computes the Bayesian score component for the given target variable index and
    Dirichlet prior counts given in alpha

INPUT:
    i       - index of the target variable
    parents - list of indeces of parent variables (should not contain self)
    r       - list of instantiation counts accessed by variable index
              r[1] gives number of discrete states variable 1 can take on
    data - matrix of sufficient statistics / counts
              d[j,k] gives the number of times the target variable took on its kth instantiation
              given the jth parental instantiation

OUTPUT:
    the Bayesian score, Float64
"""
function bayesian_score_component{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    alpha::AbstractMatrix{Float64}, # ncategories[i]×prod(ncategories[parents])
    )

    (n, m) = size(data)
    if !isempty(parents)
        Np = length(parents)
        stridevec = fill(1, Np)
        for k in 2:Np
            stridevec[k] = stridevec[k-1] * ncategories[parents[k-1]]
        end
        js = (data[parents,:] - 1)' * stridevec + 1
    else
        js = fill(1, m)
    end

    N = sparse(vec(data[i,:]), vec(js), 1, size(alpha)...) # note: duplicates are added together
    sum(lgamma.(alpha + N)) - sum(lgamma.(alpha)) + sum(lgamma.(sum(alpha,1))) - sum(lgamma.(sum(alpha,1) + sum(N,1)))::Float64
end
function bayesian_score_component_uniform{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    prior::DirichletPrior,
    )

    (n, m) = size(data)
    if !isempty(parents)
        Np = length(parents)
        stridevec = fill(1, Np)
        for k in 2:Np
            stridevec[k] = stridevec[k-1] * ncategories[parents[k-1]]
        end
        js = (data[parents,:] - 1)' * stridevec + 1
    else
        js = fill(1, m)
    end

    n = prod(ncategories[parents])
    N = sparse(data[i,:], js, 1, ncategories[i], n...) # note: duplicates are added together

    u = prior.α
    p = lgamma(u)

    # Given a sparse N, we can be clever in our calculation and not waste time
    # computing the same lgamma values by exploiting the sparse structure.
    sum0 = sum(lgamma.(nonzeros(N) + u)) + p * (ncategories[i] * n - nnz(N))
    sum1 = n * ncategories[i] * p
    sum2 = n * lgamma(ncategories[i] * u)
    cc = ncategories[i] * u

    @static if Base.VERSION.major == 0 && Base.VERSION.minor < 5
        sN = sparsevec(sum(N[1:ncategories[i],:], 1)) # Slower, but should be supported by Julia 0.4
    else
        sN = sum(N[i,:] for i=1:ncategories[i])
    end
    sum3 = sum(lgamma.(nonzeros(sN) + cc)) + (size(N, 2) - nnz(sN)) * lgamma(cc)
    sum0 - sum1 + sum2 - sum3::Float64
end
function bayesian_score_component{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    prior::DirichletPrior,
    )

    if typeof(prior) == UniformPrior
        return bayesian_score_component_uniform(i, parents, ncategories, data, prior)
    end
    alpha = get(prior, i, ncategories, parents)
    bayesian_score_component(i, parents, ncategories, data, alpha)
end

function bayesian_score(
    parent_list::Vector{Vector{Int}},
    ncategories::AbstractVector{Int},
    data::Matrix{Int},
    prior::DirichletPrior,
    )

    tot = 0.0
    for (i, p) in enumerate(parent_list)
        tot += bayesian_score_component(i, p, ncategories, data, prior)
    end
    tot
end
function bayesian_score(bn::DiscreteBayesNet, data::DataFrame, prior::DirichletPrior=UniformPrior())

    n = length(bn)
    parent_list = Array{Vector{Int}}(n)
    ncategories = Array{Int}(n)
    datamat = convert(Matrix{Int}, data)'

    for (i,cpd) in enumerate(bn.cpds)
        parent_list[i] = in_neighbors(bn.dag, i)
        ncategories[i] = infer_number_of_instantiations(convert(Vector{Int}, data[i]))
    end

    bayesian_score(parent_list, ncategories, datamat, prior)
end

function bayesian_score_component(
    i::Int,
    parents::AbstractVector{Int},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    prior::DirichletPrior,
    cache::ScoreComponentCache,
    )

    if !haskey(cache[i], parents)
        (cache[i][parents] = bayesian_score_component(i, parents, ncategories, data, prior))
    end

    cache[i][parents]
end
function bayesian_score_components(
    parent_list::Vector{Vector{Int}},
    ncategories::AbstractVector{Int},
    data::Matrix{Int},
    prior::DirichletPrior,
    )

    score_components = Array{Float64}(length(parent_list))
    for (i,p) in enumerate(parent_list)
        score_components[i] = bayesian_score_component(i, p, ncategories, data, prior)
    end
    score_components
end
function bayesian_score_components(
    parent_list::Vector{Vector{Int}},
    ncategories::AbstractVector{Int},
    data::Matrix{Int},
    prior::DirichletPrior,
    cache::ScoreComponentCache,
    )

    score_components = Array{Float64}(length(parent_list))
    for (i,p) in enumerate(parent_list)
        score_components[i] = bayesian_score_component(i, p, ncategories, data, prior, cache)
    end
    score_components
end
function bayesian_score_components(bn::DiscreteBayesNet, data::DataFrame, prior::DirichletPrior=UniformPrior())

    n = length(bn)
    parent_list = Array{Vector{Int}}(n)
    ncategories = Array{Int}(n)
    datamat = convert(Matrix{Int}, data)'

    for (i,cpd) in enumerate(bn.cpds)
        parent_list[i] = in_neighbors(bn.dag, i)
        ncategories[i] = infer_number_of_instantiations(convert(Vector{Int}, data[i]))
    end

    bayesian_score_components(parent_list, ncategories, datamat, prior)
end

"""
    bayesian_score(G::DAG, names::Vector{Symbol}, data::DataFrame[, ncategories::Vector{Int}[, prior::DirichletPrior]])

Compute the bayesian score for graph structure `g`, with the data in `data`. `names` containes a symbol corresponding to each vertex in `g` that is the name of a column in `data`. `ncategories` is a vector of the number of values that each variable in the Bayesian network can take.

Note that every entry in data must be an integer greater than 0
"""
function bayesian_score(G::DAG,
                        names::Vector{Symbol},
                        data::DataFrame,
                        ncategories::Vector{Int}=Int[infer_number_of_instantiations(convert(Vector{Int}, data[n])) for n in names],
                        prior::DirichletPrior=UniformPrior())
    datamat = Array{Int}(ncol(data), nrow(data))
    for i in 1:nv(G)
        datamat[i,:] = data[names[i]]
    end

    # NOTE: this is badj(G) prior to v0.6 and in_neighbors(G) in v0.6
    backwards_adjacency = [in_neighbors(G, i) for i in 1 : nv(G)]
    return bayesian_score(backwards_adjacency, ncategories, datamat, prior)
end
