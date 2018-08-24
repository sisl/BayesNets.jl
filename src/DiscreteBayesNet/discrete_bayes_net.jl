export
    DiscreteBayesNet

"""
`DiscreteBayesNet`s are Bayesian Networks where every variable is an integer
within 1:Nᵢ and every distribution is Categorical.

This representation is very common, and allows for the use of factors, for
example in _Probabilistic Graphical Models_ by Koller and Friedman
"""
const DiscreteBayesNet = BayesNet{DiscreteCPD}
DiscreteBayesNet() = BayesNet(DiscreteCPD)

function _get_parental_ncategories(bn::DiscreteBayesNet, parents::NodeNames)
    parental_ncategories = Array{Int}(undef, length(parents))
    for (i,p) in enumerate(parents)
        parent_cpd = get(bn, p)::CategoricalCPD

        # assumes all distributions in cpd have same num categories
        dist = parent_cpd.distributions[1]
        parental_ncategories[i] = Distributions.ncategories(parent_cpd.distributions[1])
    end

    parental_ncategories
end

"""
    rand_cpd(bn::DiscreteBayesNet, ncategories::Int, target::NodeName, parents::NodeNames=NodeName[])
Return a CategoricalCPD with the given number of categories with random categorical distributions
"""
function rand_cpd(bn::DiscreteBayesNet, ncategories::Int, target::NodeName, parents::NodeNames=NodeName[];
    uniform_dirichlet_prior::Float64 = 1.0
    )

    !haskey(bn.name_to_index, target) || error("A CPD with name $target already exists!")

    parental_ncategories = _get_parental_ncategories(bn, parents)

    Q = prod(parental_ncategories)
    distributions = Array{Categorical{Float64}}(undef, Q)
    dir = Dirichlet(ncategories, uniform_dirichlet_prior) # draw random categoricals from a Dirichlet distribution
    for q in 1:Q
        distributions[q] = Categorical{Float64}(rand(dir))
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

    return Table(d)
end

table(bn::DiscreteBayesNet, name::NodeName, a::Assignment) = partialsort(table(bn, name), a)
table(bn::DiscreteBayesNet, name::NodeName, pair::Pair{NodeName}...) =
        table(bn, name, Assignment(pair))

"""
    Distributions.ncategories(bn::DiscreteBayesNet, node::Symbol)

Return the number of categories for a node in the network.
"""
function Distributions.ncategories(bn::DiscreteBayesNet, node::NodeName)
    return ncategories(get(bn, node).distributions[1])
end

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

    return tu
end

Base.count(bn::DiscreteBayesNet, data::DataFrame) =
        map(nodename->count(bn, nodename, data), names(bn))

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
        js = (data[parents,:] .- 1)' * stridevec .+ 1
    else
        js = fill(1, size(data,2))
    end

    Matrix(sparse(vec(data[targetind,:]), vec(js), 1, ncategories[targetind], q))
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

This function uses sparse matrix black magic and was mercilessly stolen from Ed Schmerling.
"""
function statistics(
    parent_list::Vector{Vector{Int}},
    ncategories::AbstractVector{Int},
    data::AbstractMatrix{Int},
    )

    n, m = size(data)
    N = Array{Matrix{Int}}(undef, n)
    for i in 1 : n
        N[i] = statistics(i, parent_list[i], ncategories, data)
    end
    N
end

function statistics(dag::DAG, data::DataFrame)

    n = nv(dag)

    n == ncol(data) || throw(DimensionMismatch("statistics' dag and data must be of the same dimension, $n ≠ $(ncol(data))"))

    parents = [inneighbors(dag, i) for i in 1:n]
    ncategories = [Int(infer_number_of_instantiations(data[i])) for i in 1 : n]
    datamat = convert(Matrix{Int}, data)'

    statistics(parents, ncategories, datamat)
end

function statistics(bn::DiscreteBayesNet, target::NodeName, data::DataFrame)

    n = nv(bn.dag)
    targetind = bn.name_to_index[target]
    parents = inneighbors(bn.dag, targetind)
    ncategories = [Int(infer_number_of_instantiations(data[i])) for i in 1 : n]
    datamat = convert(Matrix{Int}, data)'

    statistics(targetind, parents, ncategories, datamat)
end

statistics(bn::DiscreteBayesNet, data::DataFrame) = statistics(bn.dag, data)
