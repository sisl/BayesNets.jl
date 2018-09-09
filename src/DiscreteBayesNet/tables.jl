#=
DataFrames are used to represent factors
https://en.wikipedia.org/wiki/Factor_graph

    :p is the column containing the probabilities, ::Float64
    Each variable has its own column corresponding to its assignments and named with its name

These can be obtained using the table() function
=#

mutable struct Table
    potential::DataFrame
end

Base.convert(::Type{DataFrame}, t::Table) = t.potential

for f in [:names, :unique, :size, :eltype, :setindex!, :getindex]
    @eval (Base.$f)(t::Table, x...) = $f(t.potential, x...)
end

for s in [:(==), :(!=)]
    @eval (Base.$s)(t::Table, f::DataFrame) = $s(t.potential, f)
    @eval (Base.$s)(f::DataFrame, t::Table) = $s(f, t.potential)
    @eval (Base.$s)(t1::Table, t2::Table) = $s(t1.potential, t2.potential)
end

DataFrames.nrow(t::Table) = nrow(t.potential)

"""
Table multiplication
"""
function Base.:*(t1::Table, t2::Table)
    f1 =t1.potential
    f2 =t2.potential

    onnames = setdiff(intersect(names(f1), names(f2)), [:p])
    finalnames = vcat(setdiff(union(names(f1), names(f2)), [:p]), :p)

    if isempty(onnames)
        j = join(f1, f2, kind=:cross)
    else
        j = join(f1, f2, on=onnames, kind=:outer)
    end

    j[:p] = broadcast(*, j[:p], j[:p_1])

    return Table(j[finalnames])
end

"""
    sumout(t, v)

Table marginalization
"""
function sumout(t::Table, v::NodeNameUnion)
    f = t.potential

    # vcat works for single values and vectors alike (magic?)
    remainingvars = setdiff(names(f), vcat(v, :p))

    if isempty(remainingvars)
        # they want to remove all variables except for prob column
        # uh ... 'singleton' table?
        return Table(DataFrame(p = sum(f[:p])))
    else
        # note that this will fail miserably if f is too large (~1E4 maybe?)
        #  nothing I can do; there is a github issue
        return Table(by(f, remainingvars, df -> DataFrame(p = sum(df[:p]))))
    end
end

"""
Table normalization
Ensures that the `:p` column sums to one
"""
function LinearAlgebra.normalize!(t::Table)
    t.potential[:p] ./= sum(t.potential[:p])

    return t
end

LinearAlgebra.normalize(t::Table) = normalize!(deepcopy(t))

"""
Given a Table, extract the rows which match the given assignment
"""
function Base.partialsort(t::Table, a::Assignment)
    f = t.potential

    commonNames = intersect(names(f), keys(a))
    mask = trues(size(f, 1))
    for s in commonNames
        # mask &= (f[s] .== a[s])
        vals = (f[s] .== a[s])
        for (i,v) in enumerate(vals)
            mask[i] = mask[i] & v
        end
    end

    return Table(f[mask, :])
end

"""
takes a list of observations of assignments represented as a DataFrame
or a set of data samples (without :p),
takes the unique assignments,
and estimates the associated probability of each assignment
based on its frequency of occurrence.
"""
function Distributions.fit(::Type{Table}, f::DataFrame)
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

    return Table(tu)
end

# """
# TODO: what is this for?
# """
# function estimate_convergence(t::Table, a::Assignment)
#     f = t.potential
#
#     n = size(f, 1)
#     p = zeros(n)
#     w = ones(n)
#     if haskey(f, :p)
#         w = f[:p]
#     end
#
#     dfindex = find([haskey(a, n) for n in names(f)])
#     dfvalues = [a[n] for n in names(f)[dfindex]]'
#
#     cumWeight = 0.0
#     cumTotalWeight = 0.0
#     for i in 1:n
#         if convert(Array, f[i, dfindex]) == dfvalues
#             cumWeight += w[i]
#         end
#         cumTotalWeight += w[i]
#         p[i] = cumWeight / cumTotalWeight
#     end
#
#     return p
# end
