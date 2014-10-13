export prior, logBayesScore, indexData, statistics, statistics!

function Base.count(b::BayesNet, name::NodeName, d::DataFrame)
  # find relevant variable names based on structure of network
  varnames = push!(parents(b, name), name)
  t = d[:, varnames]
  tu = unique(t)
  # add column with counts of unique samples
  tu[:count] = Int[sum([Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]]) for j = 1:size(tu,1)]
  tu
end

Base.count(b::BayesNet, d::DataFrame) = [count(b, name, d) for name in b.names]

function indexData(b::BayesNet, d::DataFrame)
    d = d[:, b.names]
    n = length(b.names)
    data = Array(Int, size(d,2), size(d,1))
    for i = 1:n
        node = b.names[i]
        elements = domain(b, node).elements
        m = [elements[i]=>i for i = 1:length(elements)]
        for j = 1:size(d, 1)
            data[i,j] = m[d[j,i]]
        end
    end
    data
end

statistics(b::BayesNet, d::DataFrame) = statistics(b, indexData(b, d))

function statistics(b::BayesNet, d::Matrix{Int})
    N = statistics(b)
    statistics!(N, b, d)
    N
end

function statistics(b::BayesNet, alpha = 0.)
    n = length(b.names)
    r = [length(domain(b, node).elements) for node in b.names]
    parentList = [int(collect(in_neighbors(i, b.dag))) for i = 1:n]
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

function statistics!(N::Vector{Any}, b::BayesNet, d::Matrix{Int})
    r = [length(domain(b, node).elements) for node in b.names]
    (n, m) = size(d)
    parentList = [int(collect(in_neighbors(i, b.dag))) for i = 1:n]
    for di = 1:m
        for i = 1:n
            k = d[i,di]
            j = 1
            p = parentList[i]
            if !isempty(p)
                ndims = length(r[p])
                j = int(d[p,di][1])
                stride = 1
                for kk=2:ndims
                    stride = stride * r[p][kk-1]
                    j += (int(d[p,di][kk])-1) * stride
                end
            end
            N[i][k,j] += 1.
        end
    end
    N
end

prior(b::BayesNet, alpha = 1.) = statistics(b::BayesNet, alpha)

function logBayesScore(N::Vector{Any}, alpha::Vector{Any})
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

function logBayesScore(b::BayesNet, d::DataFrame, alpha = 1.)
    alpha = prior(b)
    N = statistics(b, d)
    logBayesScore(N, alpha)
end
