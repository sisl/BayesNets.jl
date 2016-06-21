"""
Overwrites assignment with a joint sample from bn
    NOTE: this will condition as it goes
"""
function Base.rand!(a::Assignment, bn::BayesNet)
    for cpd in bn.cpds
        a[name(cpd)] = rand(cpd, a)
    end
    a
end
Base.rand(bn::BayesNet) = rand!(Assignment(), bn)

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows.
"""
function Base.rand(bn::BayesNet, nsamples::Integer)

    a = Assignment()
    df = DataFrame()
    for cpd in bn.cpds
        df[name(cpd)] = Array(eltype(cpd), nsamples)
    end

    for i in 1:nsamples
        rand!(a, bn)
        for cpd in bn.cpds
            n = name(cpd)
            df[i, n] = a[n]
        end
    end

    df
end

"""
Generates a DataFrame containing a dataset of variable assignments.
Always return a DataFrame with `nsamples` rows or errors out

nsamples: the number of rows the resulting DataFrame will contain
consistent_with: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1)
max_nsamples: an upper limit on the number of samples that will be tried, needed to ensure zero-prob samples are never used
"""
function Base.rand(bn::BayesNet, nsamples::Integer, consistent_with::Assignment, max_nsamples::Integer=nsamples*100)

    a = Assignment()
    df = DataFrame()
    for cpd in bn.cpds
        df[name(cpd)] = Array(eltype(cpd), nsamples)
    end

    sample_count = 0
    for i in 1:nsamples

        while sample_count < max_nsamples
            sample_count += 1

            rand!(a, bn)
            if consistent(a, consistent_with)
                break
            end
        end

        sample_count ≤ max_nsamples || error("rand hit sample threshold of $max_nsamples")

        for cpd in bn.cpds
            n = name(cpd)
            df[i, n] = a[n]
        end
    end

    df
end

# function rand_table_weighted(bn::BayesNet; numSamples::Integer=10, consistentWith::Assignment=Assignment())
#     ordering = topological_sort_by_dfs(bn.dag)
#     t = Dict([node.name => Any[] for node in bn.nodes])
#     w = ones(numSamples)
#     a = Assignment()
#     for i in 1:numSamples
#         for node in bn.nodes[ordering]
#             name = node.name
#             if haskey(consistentWith, name)
#                 a[name] = consistentWith[name]
#                 w[i] *= pdf(cpd(bn, name), a)(a[name])
#             else
#                 a[name] = rand(cpd(bn, name), a)
#             end
#             push!(t[name], a[name])
#         end
#     end
#     t[:p] = w / sum(w)
#     convert(DataFrame, t)
# end

# """
# construct a random dictionary for a Bernoulli CPD
# NOTE: entries are [0,1] and not [true, false]
# """
# function rand_bernoulli_dict(numParents::Integer)
#     dims = ntuple(i->2, numParents)
#     Dict{Vector{Int}, Float64}([[ind2sub(dims, i)...] .- 1 => round(1+rand()*98)/100 for i = 1:prod(dims)])
# end

# """
# construct a random dictionary for a discrete CPD
# """
# function rand_discrete_dict{V<:AbstractVector}(parentDomains::AbstractVector{V}, dimNode::Integer)
#     dims = ntuple(i -> length(parentDomains[i]), length(parentDomains))

#     if isempty(dims)
#         return [_normalize_values(rand(dimNode))]
#     else
#         return Dict([[_map_names(ind2sub(dims, i), parentDomains)...] => _normalize_values(rand(dimNode)) for i = 1:prod(dims)])
#     end
# end

# _map_names(dim, names) = ntuple(i -> names[i][dim[i]], length(dim))
# _normalize_values(d) = d /= sum(d)