export prior, logBayesScore, indexData, statistics, statistics!

function Base.count(bn::BayesNet, name::NodeName, d::DataFrame)
  # find relevant variable names based on structure of network
  varnames = push!(parents(bn, name), name)
  t = d[:, varnames]
  tu = unique(t)
  # add column with counts of unique samples
  tu[:count] = Int[sum(Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]) for j = 1:size(tu,1)]
  tu
end

Base.count(bn::BayesNet, d::DataFrame) = [count(bn, node.name, d) for node in bn.nodes]

function indexData(bn::BayesNet, d::DataFrame)
    d = d[:, names(bn)]
    n = length(bn.nodes)
    data = Array(Int, size(d,2), size(d,1))
    for (i,node) in enumerate(bn.nodes)
        name = node.name
        elements = domain(bn, name).elements
        m = Dict([elements[i]=>i for i = 1:length(elements)])
        for j = 1:size(d, 1)
            data[i,j] = m[d[j,i]]
        end
    end
    data
end

statistics(bn::BayesNet, d::DataFrame) = statistics(bn, indexData(bn, d))

function statistics(bn::BayesNet, d::Matrix{Int})
    N = statistics(bn)
    statistics!(N, bn, d)
    N
end

function statistics(bn::BayesNet, alpha::Float64 = 0.0)
    n = length(bn.nodes)
    r = [length(domain(bn, node.name).elements) for node in bn.nodes]
    parentList = [collect(in_neighbors(bn.dag, i)) for i = 1:n]
    N = cell(n)
    for i = 1:n
        q = 1
        if !isempty(parentList[i])
            q = prod(r[parentList[i]])
        end
        N[i] = ones(r[i], q) * alpha
    end
    N
end

function statistics!(N::Vector{Any}, bn::BayesNet, d::Matrix{Int})
    r = [length(domain(bn, node.name).elements) for node in bn.nodes]
    (n, m) = size(d)
    parentList = [collect(in_neighbors(bn.dag, i)) for i = 1:n]
    for i = 1:n
        p = parentList[i]
        if !isempty(p)
            Np = length(p)
            stridevec = fill(1, length(p))
            for k = 2:Np
                stridevec[k] = stridevec[k-1] * r[p[k-1]]
            end
            js = d[p,:]' * stridevec - sum(stridevec) + 1
            # side note: flipping d to make array access column-major improves speed by a further 10%
            # this change could be hacked into this method (dT = d'), but should really be made in indexData
        else
            js = fill(1, m)
        end
        N[i] += sparse(vec(d[i,:]), vec(js), 1, size(N[i])...)
    end
    N
end

prior(bn::BayesNet, alpha::Real = 1.0) = statistics(bn, alpha)

function log_bayes_score(N::Vector{Any}, alpha::Vector{Any})
    @assert length(N) == length(alpha)
    n = length(N)
    p = 0.
    for i = 1:n
        if !isempty(N[i])
            p += sum(lgamma(alpha[i] + N[i]))
            p -= sum(lgamma(alpha[i]))
            p += sum(lgamma(sum(alpha[i],1)))
            p -= sum(lgamma(sum(alpha[i],1) + sum(N[i],1)))
        end
    end
    p
end
function log_bayes_score(bn::BayesNet, d::Union{DataFrame, Matrix{Int}}, alpha::Real = 1.0)
    alpha = prior(bn)
    N = statistics(bn, d)
    log_bayes_score(N, alpha)
end
