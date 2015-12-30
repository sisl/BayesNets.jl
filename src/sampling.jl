function Base.rand(b::BayesNet)
  ordering = topological_sort_by_dfs(b.dag)
  a = Assignment()
  for node in b.nodes[ordering]
    name = node.name
    a[name] = rand(cpd(b, name), a)
  end
  a
end

function randTable(b::BayesNet; numSamples=10, consistentWith=Assignment())
    ordering = topological_sort_by_dfs(b.dag)
    t = Dict([node.name => Any[] for node in b.nodes])
    a = Assignment()

    for i in 1:numSamples
        for node in b.nodes[ordering]
            name = node.name
            a[name] = rand(cpd(b, name), a)
        end
        if consistent(a, consistentWith)
            for node in b.nodes[ordering]
                name = node.name
                push!(t[name], a[name])
            end
        end
    end
    convert(DataFrame, t)
end

function randTableWeighted(b::BayesNet; numSamples=10, consistentWith=Assignment())
    ordering = topological_sort_by_dfs(b.dag)
    t = Dict([node.name => Any[] for node in b.nodes])
    w = ones(numSamples)
    a = Assignment()
    for i in 1:numSamples
        for node in b.nodes[ordering]
            name = node.name
            if haskey(consistentWith, name)
                a[name] = consistentWith[name]
                w[i] *= pdf(cpd(b, name), a)(a[name])
            else
                a[name] = rand(cpd(b, name), a)
            end
            push!(t[name], a[name])
        end
    end
    t[:p] = w / sum(w)
    convert(DataFrame, t)
end

# generate a random dictionary of Bernoulli parameters
# given some number of parents
function randBernoulliDict(numParents::Integer)
    dims = ntuple(i->2, numParents)
    Dict([[ind2sub(dims, i)...] .- 1 => round(1+rand()*98)/100 for i = 1:prod(dims)])
end

function normalize_values(d)
    return d /= sum(d)
end

function map_names(dim, names)
    return ntuple(length(dim), i -> names[i][dim[i]])
end

function randDiscreteDict(dimParents, dimNode)
    dims = ntuple(length(dimParents), i -> length(dimParents[i]))

    if length(dims) == 0
        return [normalize_values(rand(dimNode))]
    else
        return Dict([[map_names(ind2sub(dims, i), dimParents)...] => normalize_values(rand(dimNode)) for i = 1:prod(dims)])
    end
end
