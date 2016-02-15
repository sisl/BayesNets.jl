function Base.rand(bn::BayesNet)
    ordering = topological_sort_by_dfs(bn.dag)
    a = Assignment()
    for node in bn.nodes[ordering]
        name = node.name
        my_cpd = cpd(bn, name)
        a[name] = rand(my_cpd, a)
    end
    a
end

Distributions.pdf(bn::BayesNet, name::NodeName, a::Assignment) = pdf(cpd(bn, name), a, parents(bn, name))

# function rand_table(bn::BayesNet; numSamples::Integer=10, consistentWith::Assignment=Assignment())
#     ordering = topological_sort_by_dfs(bn.dag)
#     t = Dict([node.name => Any[] for node in bn.nodes])
#     a = Assignment()

#     for i in 1:numSamples
#         for node in bn.nodes[ordering]
#             name = node.name
#             a[name] = rand(cpd(bn, name), a)
#         end
#         if consistent(a, consistentWith)
#             for node in bn.nodes[ordering]
#                 name = node.name
#                 push!(t[name], a[name])
#             end
#         end
#     end
#     convert(DataFrame, t)
# end

# """
# Generates a DataFrame containing a dataset of variable assignments.
# Should always return a DataFrame with `numSamples` rows.

# numSamples: the number of rows the resulting DataFrame will contain
# consistentWith: the assignment that all samples must be consistent with (ie, Dict[:A=>1] means all samples must have :A==1)
# maxNumSamples: an upper limit on the number of samples that will be tried, needed to ensure zero-prob samples are never used
# """
# function rand_table(bn::BayesNet; numSamples::Integer=10, consistentWith::Assignment=Assignment(), maxNumSamples::Integer=numSamples*100)
#     ordering = topological_sort_by_dfs(bn.dag)
#     t = Dict([node.name => Any[] for node in bn.nodes])
#     a = Assignment()

#     for i in 1:numSamples
#         for node in bn.nodes[ordering]
#             name = node.name
#             a[name] = get(consistentWith, name, rand(cpd(bn, name), a))
#            # TODO: check that resulting CPD is non-zero


#             push!(t[name], a[name])
#         end
#     end
#     convert(DataFrame, t)
# end

function rand_table_weighted(bn::BayesNet; numSamples::Integer=10, consistentWith::Assignment=Assignment())
    ordering = topological_sort_by_dfs(bn.dag)
    t = Dict([node.name => Any[] for node in bn.nodes])
    w = ones(numSamples)
    a = Assignment()
    for i in 1:numSamples
        for node in bn.nodes[ordering]
            name = node.name
            if haskey(consistentWith, name)
                a[name] = consistentWith[name]
                w[i] *= pdf(cpd(bn, name), a)(a[name])
            else
                a[name] = rand(cpd(bn, name), a)
            end
            push!(t[name], a[name])
        end
    end
    t[:p] = w / sum(w)
    convert(DataFrame, t)
end

"""
construct a random dictionary for a Bernoulli CPD
NOTE: entries are [0,1] and not [true, false]
"""
function rand_bernoulli_dict(numParents::Integer)
    dims = ntuple(i->2, numParents)
    Dict{Vector{Int}, Float64}([[ind2sub(dims, i)...] .- 1 => round(1+rand()*98)/100 for i = 1:prod(dims)])
end

"""
construct a random dictionary for a discrete CPD
"""
function rand_discrete_dict{V<:AbstractVector}(parentDomains::AbstractVector{V}, dimNode::Integer)
    dims = ntuple(i -> length(parentDomains[i]), length(parentDomains))

    if isempty(dims)
        return [_normalize_values(rand(dimNode))]
    else
        return Dict([[_map_names(ind2sub(dims, i), parentDomains)...] => _normalize_values(rand(dimNode)) for i = 1:prod(dims)])
    end
end

_map_names(dim, names) = ntuple(i -> names[i][dim[i]], length(dim))
_normalize_values(d) = d /= sum(d)