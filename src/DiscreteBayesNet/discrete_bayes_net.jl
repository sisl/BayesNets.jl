export
    DiscreteBayesNet

"""
`DiscreteBayesNet`s are Bayesian Networks where every variable is an integer
within 1:Nᵢ and every distribution is Categorical.

This representation is very common, and allows for the use of factors, for
example in _Probabilistic Graphical Models_ by Koller and Friedman
"""
typealias DiscreteBayesNet BayesNet{DiscreteCPD}
DiscreteBayesNet() = BayesNet(DiscreteCPD)

function _get_parental_ncategories(bn::DiscreteBayesNet, parents::Vector{NodeName})
    parental_ncategories = Array(Int, length(parents))
    for (i,p) in enumerate(parents)
        parent_cpd = get(bn, p)::CategoricalCPD

        # assumes all distributions in cpd have same num categories
        dist = parent_cpd.distributions[1]
        parental_ncategories[i] = Distributions.ncategories(parent_cpd.distributions[1])
    end

    parental_ncategories
end

"""
    rand_cpd(bn::DiscreteBayesNet, ncategories::Int, target::NodeName, parents::Vector{NodeName}=NodeName[])
Return a CategoricalCPD with the given number of categories with random categorical distributions
"""
function rand_cpd(bn::DiscreteBayesNet, ncategories::Int, target::NodeName, parents::Vector{NodeName}=NodeName[];
    uniform_dirichlet_prior::Float64 = 1.0
    )

    !haskey(bn.name_to_index, target) || error("A CPD with name $target already exists!")

    parental_ncategories = _get_parental_ncategories(bn, parents)

    Q = prod(parental_ncategories)
    distributions = Array(Categorical, Q)
    dir = Dirichlet(ncategories, uniform_dirichlet_prior) # draw random categoricals from a Dirichlet distribution
    for q in 1:Q
        distributions[q] = Categorical(rand(dir))
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end


"""
    table(bn::DiscreteBayesNet, name::NodeName)
Constructs the CPD factor associated with the given node in the BayesNet
"""
function table(bn::DiscreteBayesNet, name::NodeName)

    d = DataFrame()
    cpd = get(bn, name)
    varnames = push!(deepcopy(parents(bn, name)), name)

    nparents = length(varnames)-1
    assignment = Assignment()
    for _name in names(bn)
        assignment[_name] = 1
    end

    if nparents > 0
        A = ndgrid([1:ncategories(get(bn, name)(assignment)) for name in varnames]...)
        for (i,name2) in enumerate(varnames)
            d[name2] = vec(A[i])
        end
    else
        d[name] = 1:ncategories(cpd(assignment))
    end

    p = ones(size(d,1)) # the probability column
    for i in 1:size(d,1)
        assignment = Assignment()
        for j in 1:length(varnames)
            assignment[varnames[j]] = d[i,j]
        end
        p[i] = pdf(cpd, assignment)
    end
    d[:p] = p
    d
end

table(bn::DiscreteBayesNet, name::NodeName, a::Assignment) = select(table(bn, name), a)
table(bn::DiscreteBayesNet, name::NodeName, pair::Pair{NodeName}...) = table(bn, name, Assignment(pair))

"""
    Base.count(bn::BayesNet, name::NodeName, data::DataFrame)
returns a table containing all observed assignments and their
corresponding counts
"""
function Base.count(bn::DiscreteBayesNet, name::NodeName, data::DataFrame)

    # find relevant variable names based on structure of network
    varnames = push!(deepcopy(parents(bn, name)), name)

    t = data[:, varnames]
    tu = unique(t)

    # add column with counts of unique samples
    tu[:count] = Int[sum(Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]) for j = 1:size(tu,1)]
    tu
end
Base.count(bn::DiscreteBayesNet, data::DataFrame) = map(nodename->count(bn, nodename, data), names(bn))


"""
    statistics(
        targetind::Int,
        parents::AbstractVector{Int},
        ncategories::AbstractVector{Int},
        data::AbstractMatrix{Int}
        )
outputs a sufficient statistics table for the target variable
that is r × q where
r = ncategories[i] is the number of variable instantiations and
q is the number of parental instantiations of variable i

The r-values are ordered from 1 → ncategories[i]
The q-values are ordered in the same ordering as ind2sub() in Julia Base
    Thus the instantiation of the first parent (by order given in parents[i])
    is varied the fastest.

ex:
    Variable 1 has parents 2 and 3, with r₁ = 2, r₂ = 2, r₃ = 3
    q for variable 1 is q = r₂×r₃ = 6
    N will be a 6×2 matrix where:
        N[1,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 1
        N[2,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 1
        N[3,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 2
        N[4,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 2
        N[5,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 3
        N[6,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 3
        N[6,2] is the number of time v₁ = 2, v₂ = 1, v₃ = 1
        ...
"""
function statistics(
    targetind::Int,
    parents::AbstractVector{Int},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int}
    )

    q = 1
    if !isempty(parents)
        Np = length(parents)
        q  = prod(ncategories[parents])
        stridevec = fill(1, Np)
        for k in 2:Np
            stridevec[k] = stridevec[k-1] * ncategories[parents[k-1]]
        end
        js = (data[parents,:] - 1)' * stridevec + 1
    else
        js = fill(1, size(data,2))
    end
    full(sparse(vec(data[targetind,:]), vec(js), 1, ncategories[targetind], q))
end

"""
    statistics(
        parent_list::Vector{Vector{Int}},
        ncategories::AbstractVector{Int},
        data::AbstractMatrix{Int},
        )
Computes sufficient statistics from a discrete dataset
for a Discrete Bayesian Net structure

INPUT:
    parents:
        list of lists of parent indices
        A variable with index i has ncategories[i]
        and row in data[i,:]
        No acyclicity checking is done
    ncategories:
        list of variable bin counts, or number of
        discrete values the variable can take on, v ∈ {1 : ncategories[i]}
    data:
        table of discrete values [n×m]
        where n is the number of nodes
        and m is the number of samples

OUTPUT:
    N :: Vector{Matrix{Int}}
        a sufficient statistics table for each variable
        Variable with index i has statistics table N[i],
        which is r × q where
        r = ncategories[i] is the number of variable instantiations and
        q is the number of parental instantiations of variable i

        The r-values are ordered from 1 → ncategories[i]
        The q-values are ordered in the same ordering as ind2sub() in Julia Base
            Thus the instantiation of the first parent (by order given in parents[i])
            is varied the fastest.

        ex:
            Variable 1 has parents 2 and 3, with r₁ = 2, r₂ = 2, r₃ = 3
            q for variable 1 is q = r₂×r₃ = 6
            N[1] will be a 6×2 matrix where:
                N[1][1,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 1
                N[1][2,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 1
                N[1][3,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 2
                N[1][4,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 2
                N[1][5,1] is the number of time v₁ = 1, v₂ = 1, v₃ = 3
                N[1][6,1] is the number of time v₁ = 1, v₂ = 2, v₃ = 3
                N[1][6,2] is the number of time v₁ = 2, v₂ = 1, v₃ = 1
                ...

This function uses sparse matrix black magic and was
mercilessly stolen from Ed Schmerling.
"""
function statistics(
    parent_list::Vector{Vector{Int}},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    )

    n, m = size(data)
    N = Array(Matrix{Int}, n)
    for i in 1 : n
        N[i] = statistics(i, parent_list[i], ncategories, data)
    end
    N
end
function statistics(dag::DAG, data::DataFrame)

    n = nv(dag)

    n == ncol(data) || throw(DimensionMismatch("statistics' dag and data must be of the same dimension, $n ≠ $(ncol(data))"))

    parents = [in_neighbors(dag, i) for i in 1:n]
    ncategories = [Int(infer_number_of_instantiations(data[i])) for i in 1 : n]
    datamat = convert(Matrix{Int}, data)'

    statistics(parents, ncategories, datamat)
end
statistics(bn::DiscreteBayesNet, data::DataFrame) = statistics(bn.dag, data)
function statistics(bn::DiscreteBayesNet, target::NodeName, data::DataFrame)

    n = nv(bn.dag)
    targetind = bn.name_to_index[target]
    parents = in_neighbors(bn.dag, targetind)
    ncategories = [Int(infer_number_of_instantiations(data[i])) for i in 1 : n]
    datamat = convert(Matrix{Int}, data)'

    statistics(targetind, parents, ncategories, datamat)
end

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
    sum(lgamma(alpha + N)) - sum(lgamma(alpha)) + sum(lgamma(sum(alpha,1))) - sum(lgamma(sum(alpha,1) + sum(N,1)))::Float64
end
function bayesian_score_component{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    prior::DirichletPrior,
    )

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
    parent_list = Array(Vector{Int}, n)
    ncategories = Array(Int, n)
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

    score_components = Array(Float64, length(parent_list))
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

    score_components = Array(Float64, length(parent_list))
    for (i,p) in enumerate(parent_list)
        score_components[i] = bayesian_score_component(i, p, ncategories, data, prior, cache)
    end
    score_components
end
function bayesian_score_components(bn::DiscreteBayesNet, data::DataFrame, prior::DirichletPrior=UniformPrior())

    n = length(bn)
    parent_list = Array(Vector{Int}, n)
    ncategories = Array(Int, n)
    datamat = convert(Matrix{Int}, data)'

    for (i,cpd) in enumerate(bn.cpds)
        parent_list[i] = in_neighbors(bn.dag, i)
        ncategories[i] = infer_number_of_instantiations(convert(Vector{Int}, data[i]))
    end

    bayesian_score_components(parent_list, ncategories, datamat, prior)
end

#########################

type GreedyHillClimbing <: GraphSearchStrategy
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
    ncategories::Int = infer_number_of_instantiations(data[target]),
    )

    prior_counts = get(prior, ncategories)
    for v in data[target]
        prior_counts[v] += 1.0
    end

    d = Categorical(prior_counts ./ sum(prior_counts))
    CategoricalCPD(target, NodeName[], Int[], Categorical[d])
end
function Distributions.fit(::Type{DiscreteCPD},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName},
    prior::DirichletPrior;
    parental_ncategories::Vector{Int} = map!(p->infer_number_of_instantiations(data[p]), Array(Int, length(parents)), parents),
    target_ncategories::Int = infer_number_of_instantiations(data[target]),
    )

    # with parents

    if isempty(parents)
        return fit(DiscreteCPD, data, target, ncategories=target_ncategories)
    end

    nparents = length(parents)
    dims = [1:parental_ncategories[i] for i in 1:nparents]
    distributions = Array(Categorical, prod(parental_ncategories))
    for (q, parent_instantiation) in enumerate(product(dims...))

        prior_counts = get(prior, target_ncategories)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                prior_counts[data[i, target]] += 1.0
            end
        end
        distributions[q] = Categorical(prior_counts ./ sum(prior_counts))
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end
function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame, params::GreedyHillClimbing;
    ncategories::Vector{Int} = map!(i->infer_number_of_instantiations(data[i]), Array(Int, n), 1:n),
    )

    n = ncol(data)
    parent_list = map!(i->Int[], Array(Vector{Int}, n), 1:n)
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
    cpds = Array(DiscreteCPD, n)
    varnames = names(data)
    for i in 1:n
        name = varnames[i]
        parents = varnames[parent_list[i]]
        cpds[i] = fit(DiscreteCPD, data, name, parents, params.prior, parental_ncategories=ncategories[parent_list[i]], target_ncategories=ncategories[i])
    end
    BayesNet(cpds)
end