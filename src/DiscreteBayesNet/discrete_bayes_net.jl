export
    DiscreteBayesNet

"""
`DiscreteBayesNet`s are Bayesian Networks where every variable is an integer
within 1:Nᵢ and every distribution is Categorical.

This representation is very common, and allows for the use of factors, for
example in _Probabilistic Graphical Models_ by Koller and Friedman
"""
typealias DiscreteBayesNet BayesNet{CPD{Categorical, CategoricalCPD}}

"""
TODO: rename table() to factor()?
Constructs the CPD factor associated with the given node in the BayesNet
"""
function table(bn::DiscreteBayesNet, name::NodeName)

    d = DataFrame()
    cpd = get(bn, name)
    names = push!(deepcopy(parents(bn, name)), name)

    nparents = length(names)-1
    if nparents > 0
        A = ndgrid([1:ncategories(distribution(get(bn, name))) for name in names]...)
        for (i,name2) in enumerate(names)
            d[name2] = vec(A[i])
        end
    else
        d[name] = 1:ncategories(distribution(cpd))
    end

    p = ones(size(d,1)) # the probability column
    for i in 1:size(d,1)
        assignment = Assignment([names[j]=>d[i,j] for j in 1:length(names)])
        p[i] = pdf!(cpd, assignment)
    end
    d[:p] = p
    d
end

table(bn::BayesNet, name::NodeName, a::Assignment) = select(table(bn, name), a)

"""
returns a table containing all observed assignments and their
corresponding counts
"""
function Base.count(bn::BayesNet, name::NodeName, data::DataFrame)

    # find relevant variable names based on structure of network
    varnames = push!(deepcopy(parents(bn, name)), name)

    t = data[:, varnames]
    tu = unique(t)

    # add column with counts of unique samples
    tu[:count] = Int[sum(Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]) for j = 1:size(tu,1)]
    tu
end

"""
outputs a sufficient statistics table for the target variable
that is r × q where
r = bincounts[i] is the number of variable instantiations and
q is the number of parental instantiations of variable i

The r-values are ordered from 1 → bincounts[i]
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
"""
function statistics(
    targetind::Int,
    parents::AbstractVector{Int},
    bincounts::AbstractVector{Int},
    data::AbstractMatrix{Int}
    )

    q = 1
    if !isempty(parents)
        Np = length(parents)
        q  = prod(bincounts[parents])
        stridevec = fill(1, Np)
        for k in 2:Np
            stridevec[k] = stridevec[k-1] * bincounts[parents[k-1]]
        end
        js = (data[parents,:] - 1)' * stridevec + 1
    else
        js = fill(1, size(data,2))
    end
    full(sparse(vec(data[targetind,:]), vec(js), 1, bincounts[targetind], q))
end

"""
Computes sufficient statistics from a discrete dataset
for a Discrete Bayesian Net structure

INPUT:
    parents:
        list of lists of parent indices
        A variable with index i has bincounts[i]
        and row in data[i,:]
        No acyclicity checking is done
    bincounts:
        list of variable bin counts, or number of
        discrete values the variable can take on, v ∈ {1 : bincounts[i]}
    data:
        table of discrete values [n×m]
        where n is the number of nodes
        and m is the number of samples

OUTPUT:
    N :: Vector{Matrix{Int}}
        a sufficient statistics table for each variable
        Variable with index i has statistics table N[i],
        which is r × q where
        r = bincounts[i] is the number of variable instantiations and
        q is the number of parental instantiations of variable i

        The r-values are ordered from 1 → bincounts[i]
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
    parents::Vector{Vector{Int}},
    bincounts::AbstractVector{Int},
    data::AbstractMatrix{Int},
    )

    n, m = size(data)
    N = Array(Matrix{Int}, n)
    for i in 1 : n
        N[i] = statistics(i, parents[i], bincounts, data)
    end
    N
end

function statistics(dag::DAG, data::DataFrame)

    n = nv(dag)

    n == ncol(data) || throw(DimensionMismatch("statistics' dag and data must be of the same dimension, $n ≠ $(ncol(data))"))

    parents = [in_neighbors(dag, i) for i in 1:n]
    bincounts = [Int(infer_number_of_instantiations(data[i])) for i in 1 : n]
    datamat = convert(Matrix{Int}, data)'

    statistics(parents, bincounts, datamat)
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
    bincounts::AbstractVector{Int},
    data::AbstractMatrix{Int},
    alpha::AbstractMatrix{Float64}, # bincounts[i]×prod(bincounts[parents])
    )

    (n, m) = size(data)
    if !isempty(parents)
        Np = length(parents)
        stridevec = fill(1, Np)
        for k in 2:Np
            stridevec[k] = stridevec[k-1] * bincounts[parents[k-1]]
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
    bincounts::AbstractVector{Int},
    data::AbstractMatrix{Int},
    prior::DirichletPrior,
    )

    alpha = get(prior, i, bincounts, parents)
    bayesian_score_component(i, parents, bincounts, data, alpha)
end

function bayesian_score(
    parents::Vector{Vector{Int}},
    bincounts::AbstractVector{Int},
    data::Matrix{Int},
    prior::DirichletPrior,
    )

    tot = 0.0
    for (i, p) in enumerate(parents)
        tot += bayesian_score_component(i, p, bincounts, data, prior)
    end
    tot
end
function bayesian_score(bn::DiscreteBayesNet, data::DataFrame, prior::DirichletPrior=UniformPrior())

    n = length(bn)
    parents = Array(Vector{Int}, n)
    bincounts = Array(Int, n)
    datamat = convert(Matrix{Int}, data)'

    for (i,cpd) in enumerate(bn.cpds)
        parents[i] = in_neighbors(bn.dag, i)
        bincounts[i] = infer_number_of_instantiations(data[i])
    end

    bayesian_score(parents, bincounts, datamat, prior)
end

# function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame)



# end


# function optimize_structure!(
#     modelparams::ModelParams,
#     data::Union{ModelData, BN_PreallocatedData};
#     forced::Tuple{Vector{Int}, Vector{Int}}=(Int[], Int[]), # lat, lon
#     verbosity::Integer=0,
#     max_parents::Integer=6
#     )

#     binmaps = modelparams.binmaps
#     parents_lat = deepcopy(modelparams.parents_lat)
#     parents_lon = deepcopy(modelparams.parents_lon)

#     forced_lat, forced_lon = forced
#     parents_lat = sort(unique([parents_lat; forced_lat]))
#     parents_lon = sort(unique([parents_lon; forced_lon]))

#     features = modelparams.features
#     ind_lat = modelparams.ind_lat
#     ind_lon = modelparams.ind_lon
#     binmap_lat = modelparams.binmaps[ind_lat]
#     binmap_lon = modelparams.binmaps[ind_lon]

#     n_targets = 2
#     n_indicators = length(features)-n_targets

#     chosen_lat = map(i->in(n_targets+i, parents_lat), 1:n_indicators)
#     chosen_lon = map(i->in(n_targets+i, parents_lon), 1:n_indicators)

#     score_cache_lat = Dict{Vector{Int}, Float64}()
#     score_cache_lon = Dict{Vector{Int}, Float64}()

#     α = modelparams.dirichlet_prior
#     score_lat = calc_component_score(ind_lat, parents_lat, binmap_lat, data, α, score_cache_lat)
#     score_lon = calc_component_score(ind_lon, parents_lon, binmap_lon, data, α, score_cache_lon)
#     score = score_lat + score_lon

#     if verbosity > 0
#         println("Starting Score: ", score)
#     end

#     n_iter = 0
#     score_diff = 1.0
#     while score_diff > 0.0
#         n_iter += 1

#         selected_lat = false
#         selected_index = 0
#         new_parents_lat = copy(parents_lat)
#         new_parents_lon = copy(parents_lon)
#         score_diff = 0.0

#         # check edges for indicators -> lat
#         if length(parents_lat) < max_parents
#             for i = 1 : n_indicators
#                 # add edge if it does not exist
#                 if !chosen_lat[i]
#                     new_parents = sort!(push!(copy(parents_lat), n_targets+i))
#                     new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, α, score_cache_lat) - score_lat
#                     if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                         selected_lat = true
#                         score_diff = new_score_diff
#                         new_parents_lat = new_parents
#                     end
#                 end
#             end
#         elseif verbosity > 0
#             warn("DBNB: optimize_structure: max parents lat reached")
#         end
#         for (idx, i) in enumerate(parents_lat)
#             # remove edge if it does exist
#             if !in(features[i], forced_lat)
#                 new_parents = deleteat!(copy(parents_lat), idx)
#                 new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, α, score_cache_lat) - score_lat
#                 if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                     selected_lat = true
#                     score_diff = new_score_diff
#                     new_parents_lat = new_parents
#                 end
#             end
#         end

#         # check edges for indicators -> lon
#         if length(parents_lon) < max_parents
#             for i = 1 : n_indicators
#                 # add edge if it does not exist
#                 if !chosen_lon[i]
#                     new_parents = sort!(push!(copy(parents_lon), n_targets+i))
#                     new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, α, score_cache_lon) - score_lon
#                     if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                         selected_lat = false
#                         score_diff = new_score_diff
#                         new_parents_lon = new_parents
#                     end
#                 end
#             end
#         elseif verbosity > 0
#             warn("DBNB: optimize_structure: max parents lon reached")
#         end
#         for (idx, i) in enumerate(parents_lon)
#             # remove edge if it does exist
#             if !in(features[i], forced_lon)
#                 new_parents = deleteat!(copy(parents_lon), idx)
#                 new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, α, score_cache_lon) - score_lon
#                 if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                     selected_lat = false
#                     score_diff = new_score_diff
#                     new_parents_lon = new_parents
#                 end
#             end
#         end

#         # check edge between lat <-> lon
#         if !in(ind_lon, parents_lat) && !in(ind_lat, parents_lon)
#             # lon -> lat
#             if length(parents_lat) < max_parents
#                 new_parents = unshift!(copy(parents_lat), ind_lon)
#                 new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, α, score_cache_lat) - score_lat
#                 if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                     selected_lat = true
#                     score_diff = new_score_diff
#                     new_parents_lat = new_parents
#                 end
#             end

#             # lat -> lon
#             if length(parents_lon) < max_parents
#                 new_parents = unshift!(copy(parents_lon), ind_lat)
#                 new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, α, score_cache_lon) - score_lon
#                 if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#                     selected_lat = false
#                     score_diff = new_score_diff
#                     new_parents_lon = new_parents
#                 end
#             end
#         # elseif in(ind_lon, parents_lat) && !in(features[ind_lon], forced_lat)

#         #     # try edge removal
#         #     new_parents = deleteat!(copy(parents_lat), ind_lat)
#         #     new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, α, score_cache_lat) - score_lat
#         #     if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#         #         selected_lat = true
#         #         score_diff = new_score_diff
#         #         new_parents_lat = new_parents
#         #     end

#         #     # try edge reversal (lat -> lon)
#         #     if length(parents_lon) < max_parents
#         #         new_parents = unshift!(copy(parents_lon), ind_lat)
#         #         new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, α, score_cache_lon) - score_lon
#         #         if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#         #             selected_lat = false
#         #             score_diff = new_score_diff
#         #             new_parents_lon = new_parents
#         #         end
#         #     end
#         # elseif in(ind_lat, parents_lon)  && !in(features[ind_lat], forced_lon)

#         #     # try edge removal
#         #     new_parents = deleteat!(copy(parents_lon), ind_lat)
#         #     new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, α, score_cache_lon) - score_lon
#         #     if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#         #         selected_lat = false
#         #         score_diff = new_score_diff
#         #         new_parents_lon = new_parents
#         #     end

#         #     # try edge reversal (lon -> lat)
#         #     if length(parents_lat) < max_parents
#         #         new_parents = unshift!(copy(parents_lat), ind_lon)
#         #         new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, α, score_cache_lat) - score_lat
#         #         if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
#         #             selected_lat = true
#         #             score_diff = new_score_diff
#         #             new_parents_lat = new_parents
#         #         end
#         #     end
#         end

#         # select best
#         if score_diff > 0.0
#             if selected_lat
#                 parents_lat = new_parents_lat
#                 chosen_lat = map(k->in(n_targets+k, parents_lat), 1:n_indicators)
#                 score += score_diff
#                 score_lat += score_diff
#                 if verbosity > 0
#                     println("changed lat:", map(f->symbol(f), features[parents_lat]))
#                     println("new score: ", score)
#                 end
#             else
#                 parents_lon = new_parents_lon
#                 chosen_lon = map(k->in(n_targets+k, parents_lon), 1:n_indicators)
#                 score += score_diff
#                 score_lon += score_diff
#                 if verbosity > 0
#                     println("changed lon:", map(f->symbol(f), features[parents_lon]))
#                     println("new score: ", score)
#                 end
#             end
#         end
#     end

#     empty!(modelparams.parents_lat)
#     empty!(modelparams.parents_lon)
#     append!(modelparams.parents_lat, parents_lat)
#     append!(modelparams.parents_lon, parents_lon)

#     modelparams
# end

# Base.count(bn::BayesNet, d::DataFrame) = [count(bn, node.name, d) for node in bn.nodes]

# """
# Converts a dataframe containing node assignments to a Matrix{Int}
# of node assignments, where M[i,j] is the assignment for the ith variable
# in the jth sample
# """
# function index_data(bn::BayesNet, d::DataFrame)
#     d = d[:, names(bn)]
#     n = length(bn.nodes)
#     data = Array(Int, size(d,2), size(d,1))
#     for (i,node) in enumerate(bn.nodes)
#         name = node.name
#         elements = domain(bn, name).elements
#         m = Dict([elements[i]=>i for i = 1:length(elements)])
#         for j = 1:size(d, 1)
#             data[i,j] = m[d[j,i]]
#         end
#     end
#     data
# end

# function statistics(bn::BayesNet, alpha::Float64 = 0.0)
#     n = length(bn.nodes)
#     r = [length(domain(bn, node.name).elements) for node in bn.nodes]
#     parentList = [collect(in_neighbors(bn.dag, i)) for i = 1:n]
#     N = cell(n)
#     for i = 1:n
#         q = 1
#         if !isempty(parentList[i])
#             q = prod(r[parentList[i]])
#         end
#         N[i] = ones(r[i], q) * alpha
#     end
#     N
# end
# function statistics(bn::BayesNet, d::Matrix{Int})
#     N = statistics(bn)
#     statistics!(N, bn, d)
#     N
# end
# statistics(bn::BayesNet, d::DataFrame) = statistics(bn, index_data(bn, d))

# function statistics!(N::Vector{Any}, bn::BayesNet, d::Matrix{Int})
#     r = [length(domain(bn, node.name).elements) for node in bn.nodes]
#     (n, m) = size(d)
#     parentList = [collect(in_neighbors(bn.dag, i)) for i = 1:n]
#     for i = 1:n
#         p = parentList[i]
#         if !isempty(p)
#             Np = length(p)
#             stridevec = fill(1, length(p))
#             for k = 2:Np
#                 stridevec[k] = stridevec[k-1] * r[p[k-1]]
#             end
#             js = d[p,:]' * stridevec - sum(stridevec) + 1
#             # side note: flipping d to make array access column-major improves speed by a further 10%
#             # this change could be hacked into this method (dT = d'), but should really be made in indext_data
#         else
#             js = fill(1, m)
#         end
#         N[i] += sparse(vec(d[i,:]), vec(js), 1, size(N[i])...)
#     end
#     N
# end

# prior(bn::BayesNet, alpha::Real = 1.0) = statistics(bn, alpha)

# function log_bayes_score(N::Vector{Any}, alpha::Vector{Any})
#     @assert length(N) == length(alpha)
#     n = length(N)
#     p = 0.
#     for i = 1:n
#         if !isempty(N[i])
#             p += sum(lgamma(alpha[i] + N[i]))
#             p -= sum(lgamma(alpha[i]))
#             p += sum(lgamma(sum(alpha[i],1)))
#             p -= sum(lgamma(sum(alpha[i],1) + sum(N[i],1)))
#         end
#     end
#     p
# end
# function log_bayes_score(bn::BayesNet, d::Union{DataFrame, Matrix{Int}}, alpha::Real = 1.0)
#     alpha = prior(bn)
#     N = statistics(bn, d)
#     log_bayes_score(N, alpha)
# end
