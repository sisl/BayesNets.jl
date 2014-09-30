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

function logBayesScore(b::BayesNet, d::DataFrame)

    n = length(b.names)

    DATA = Array(Any, size(d))
    for i = 1:n
        node = b.names[i]
        
        DATA[:, i] = d[node]
    end

    score = 0

    for i = 1:n
        node = b.names[i]
        node_idx = i

        pars = parents(b, node)
        pars_idx = zeros(Int64, length(pars))
        for l = 1:length(pars)
            pars_idx[l] = b.index[pars[l]]
        end
        sort!(pars_idx)

        q = 1
        for l = 1:length(pars)
            q *= length(domain(b, pars[l]).elements)
        end

        r = length(domain(b, node).elements)

        C = Dict()
        sizehint(C, size(q * r, 1))
        for l = 1:size(d, 1)
            #key = array(d[l, [pars, node]])
            key = DATA[l, [pars_idx, node_idx]]

            if !haskey(C, key)
                C[key] = 1
            else
                C[key] += 1
            end
        end

        # let alpha_ijk = 1

        second_term = 0
        D = Dict()
        sizehint(D, length(q))
        for (key, m) in C
            key_ = key[1:(end-1)]

            if !haskey(D, key_)
                D[key_] = m
            else
                D[key_] += m
            end

            second_term += lgamma(1 + m) - lgamma(1)
        end

        first_term = 0
        for (key, m0) in D
            first_term += lgamma(r) - lgamma(r + m0)
        end

        score += first_term + second_term
    end

    return score
end

