#=
DataFrames are used to represent conditional probability tables
    :p is the column containing the probability, ::Float64, must sum to 1
    Each variable has its own column corresponding to its assignments and named with its name

These can be obtained using the table() function
=#

Base.zero(::Any) = "" # TODO: DO WE NEED THIS?

"""
Multiple two conditional probability tables, represented as DataFrames, together
"""
function Base.(:(*))(df1::DataFrame, df2::DataFrame)
    onnames = setdiff(intersect(names(df1), names(df2)), [:p])
    finalnames = vcat(setdiff(union(names(df1), names(df2)), [:p]), :p)
    if isempty(onnames)
        j = join(df1, df2, kind=:cross)
        j[:,:p] .*= j[:,:p_1]
        return j[:,finalnames]
    else
        j = join(df1, df2, on=onnames, kind=:outer)
        j[:,:p] .*= j[:,:p_1]
        return j[:,finalnames]
    end
end

# TODO: this currently only supports binary-valued variables
function sumout(a::DataFrame, v::Symbol)
    @assert issubset(unique(a[:,v]), [false, true])
    remainingvars = setdiff(names(a), [v, :p])
    g = groupby(a, v)
    if length(g) == 1
        return a[:,vcat(remainingvars, :p)]
    end
    j = join(g..., on=remainingvars)
    j[:,:p] += j[:,:p_1]
    j[:,vcat(remainingvars, :p)]
end
function sumout(a::DataFrame, v::Vector{Symbol})
    if isempty(v)
        return a
    else
        sumout(sumout(a, v[1]), v[2:end])
    end
end

function normalize(a::DataFrame)
    a[:,:p] /= sum(a[:,:p])
    a
end

"""
Given a DataFrame representation of a conditional probability table,
extract the rows which match the given assignment
"""
function select(t::DataFrame, a::Assignment)
    commonNames = intersect(names(t), keys(a))
    mask = trues(size(t,1))
    for s in commonNames
        mask &= (t[s] .== a[s])
    end
    t[mask, :]
end

function estimate(df::DataFrame)
    n = size(df, 1)
    w = ones(n)
    t = df
    if haskey(df, :p)
        t = df[:, names(t) .!= :p]
        w = df[:p]
    end
    # unique samples
    tu = unique(t)
    # add column with probabilities of unique samples
    tu[:p] = Float64[sum(w[Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]]) for j = 1:size(tu,1)]
    tu[:p] /= sum(tu[:p])
    tu
end

function estimateConvergence(df::DataFrame, a::Assignment)
    n = size(df, 1)
    p = zeros(n)
    w = ones(n)
    if haskey(df, :p)
        w = df[:p]
    end
    dfindex = find([haskey(a, n) for n in names(df)])
    dfvalues = [a[n] for n in names(df)[dfindex]]'
    cumWeight = 0.
    cumTotalWeight = 0.
    for i = 1:n
        if array(df[i, dfindex]) == dfvalues
            cumWeight += w[i]
        end
        cumTotalWeight += w[i]
        p[i] = cumWeight / cumTotalWeight
    end
    p
end