#=
DataFrames are used to represent factors
https://en.wikipedia.org/wiki/Factor_graph

    :p is the column containing the probabilities, ::Float64
    Each variable has its own column corresponding to its assignments and named with its name

These can be obtained using the table() function
=#

typealias Factor DataFrame

"""
Factor multiplication
"""
@compat function Base.:*(f1::Factor, f2::Factor)
    onnames = setdiff(intersect(names(f1), names(f2)), [:p])
    finalnames = vcat(setdiff(union(names(f1), names(f2)), [:p]), :p)

    if isempty(onnames)
        j = join(f1, f2, kind=:cross)
    else
        j = join(f1, f2, on=onnames, kind=:outer)
    end

    for k in 1 : length(j[:p])
        j[k,:p] *= j[k,:p_1]
    end
    return j[:,finalnames]
end

# TODO: implement factoring out final value in factor table,
#       or throwing an error in that case
# Works for non-binary variables and possibly fixes the above todo
"""
Factor marginalization
"""
function sumout(f::Factor, v::Union{Symbol, AbstractVector{Symbol}})
    # vcat works for single values and vectors alike (magic?)
    remainingvars = setdiff(names(f), vcat(v, :p))

    if isempty(remainingvars)
        # they want to remove all variables except for prob column
        # uh ...
        return f
    else
        # note that this will fail miserablely if f is too large (~1E4 maybe?)
        #  nothing I can do :'(  github issue about it
        return by(f, remainingvars, df -> Factor(p = sum(df[:p])))
    end
end

# Should normalize be normalize! since it modifies the table?

"""
Factor normalization
Ensures that the :p column sums to one
"""
function normalize(f::Factor)
    f[:p] /= sum(f[:p])

    return f
end

"""
Given a Factor,
extract the rows which match the given assignment
"""
function Base.select(f::Factor, a::Assignment)
    commonNames = intersect(names(f), keys(a))
    mask = trues(size(f,1))
    for s in commonNames
        mask &= (f[s] .== a[s])
    end
    f[mask, :]
end

"""
takes a factor represented as a DataFrame
or a set of data samples (without :p),
takes the unique assignments,
and estimates the associated probability of each assignment
based on its frequency of occurrence.
"""
function Distributions.estimate(f::DataFrame)
    w = ones(size(f, 1))
    t = f
    if haskey(f, :p)
        t = f[:, names(t) .!= :p]
        w = f[:p]
    end
    # unique samples
    tu = unique(t)
    # add column with probabilities of unique samples
    tu[:p] = Float64[sum(w[Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]]) for j = 1:size(tu,1)]
    tu[:p] /= sum(tu[:p])
    tu
end

"""
TODO: what is this for?
"""
function estimate_convergence(f::DataFrame, a::Assignment)

    n = size(f, 1)
    p = zeros(n)
    w = ones(n)
    if haskey(f, :p)
        w = f[:p]
    end

    dfindex = find([haskey(a, n) for n in names(f)])
    dfvalues = [a[n] for n in names(f)[dfindex]]'

    cumWeight = 0.0
    cumTotalWeight = 0.0
    for i in 1:n
        if convert(Array, f[i, dfindex]) == dfvalues
            cumWeight += w[i]
        end
        cumTotalWeight += w[i]
        p[i] = cumWeight / cumTotalWeight
    end
    p
end
