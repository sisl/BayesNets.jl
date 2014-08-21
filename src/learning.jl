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

function prior(bn::BayesNet, name::NodeName, domains::Vector, alpha::Real = 1)
  edges = in_edges(bn.index[name], bn.dag)
  names = [bn.names[source(e, bn.dag)] for e in edges]
  names = [names, name]
  d = DataFrame()
  if length(edges) > 0
    A = ndgrid([domain(bn, name).elements for name in names]...)
    i = 1
    for name in names
      d[name] = A[i][:]
      i = i + 1
    end
  else
    d[name] = domain(bn, name).elements
  end
  for i = 1:size(d,1)
    ownValue = d[i,length(names)]
    a = [names[j]=>d[i,j] for j = 1:(length(names)-1)]
  end
  d[:count] = ones(size(d,1))
  d
end

prior(bn::BayesNet, name::NodeName, alpha::Real = 1) = prior(bn, name, [domain(bn, name).elements for name in bn.names], alpha)

prior(bn::BayesNet, domains::Vector, alpha::Real = 1) = [prior(bn, name, domains, alpha) for name in bn.names]

function prior(bn::BayesNet, alpha::Real = 1)
  domains = [domain(bn, name).elements for name in bn.names]
  [prior(bn, name, domains, alpha) for name in bn.names]
end

# helper for logBayesScore
function sumoutCounts(a::DataFrame, v::Symbol)
  @assert issubset(unique(a[:,v]), [false,true])
  remainingvars = setdiff(names(a), [v, :count])
  if isempty(remainingvars)
    j = DataFrame()
    j[:count] = [sum(a[:count])]
    return j
  else
    g = groupby(a, v)
    j = join(g..., on=remainingvars)
    j[:,:count] += j[:,:count_1]
    j[isna(j[:count]), :count] = 0
    return j
  end
end

# helper for logBayesScore
function addCounts(df1::DataFrame, df2::DataFrame)
  onnames = setdiff(intersect(names(df1), names(df2)), [:count])
  finalnames = vcat(setdiff(union(names(df1), names(df2)), [:count]), :count)
  if isempty(onnames)
    j = join(df1, df2, kind=:cross)
    j[isna(j[:count]), :count] = 0
    j[isna(j[:count_1]), :count_1] = 0
    j[:,:count] .+= j[:,:count_1]
    j[isna(j[:count]), :count] = 0
    return j[:,finalnames]
  else
    j = join(df1, df2, on=onnames, kind=:outer)
    j[isna(j[:count]), :count] = 0
    j[isna(j[:count_1]), :count_1] = 0
    j[:,:count] .+= j[:,:count_1]
    j[isna(j[:count]), :count] = 0
    return j[:,finalnames]
  end
end

function logBayesScore(b::BayesNet, d::DataFrame)
  n = length(b.names)
  M = count(b, d)
  Alpha = prior(b, 1)
  score = 0.
  for i = 1:n
    sym = b.names[i]
    m = M[i]
    alpha = Alpha[i]
    s = addCounts(m, alpha)
    alpha0 = sumoutCounts(alpha, sym)
    s0 = sumoutCounts(s, sym)
    score += sum(lgamma(alpha0[:count]) - lgamma(s0[:count]))
    score += sum(lgamma(s[:count]) - lgamma(alpha[:count]))
  end
  score
end

