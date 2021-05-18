#=
DataFrames are used to represent factors
https://en.wikipedia.org/wiki/Factor_graph

    :potential is the column containing the probabilities, ::Float64
    Each variable has its own column corresponding to its assignments and named with its name

These can be obtained using the table() function
=#

mutable struct Table
    potential::DataFrame
end

Base.convert(::Type{DataFrame}, t::Table) = t.potential

for f in [:names, :unique, :size, :eltype, :setindex!]
    @eval (Base.$f)(t::Table, x...) = $f(t.potential, x...)
end
Base.getindex(t::Table, x...) = getindex(t.potential, !, x...)

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

    onnames = setdiff(intersect(propertynames(f1), propertynames(f2)), [:potential])
    finalnames = vcat(setdiff(union(propertynames(f1), propertynames(f2)), [:potential]), :potential)

    if isempty(onnames)
        j = join(f1, f2, kind=:cross, makeunique=true)
    else
        j = outerjoin(f1, f2, on=onnames, makeunique=true)
    end

    j[!,:potential] = broadcast(*, j[!,:potential], j[!,:potential_1])

    return Table(j[!,finalnames])
end

"""
    sumout(t, v)

Table marginalization
"""
function sumout(t::Table, v::NodeNameUnion)
    f = t.potential

    # vcat works for single values and vectors alike (magic?)
    remainingvars = setdiff(propertynames(f), vcat(v, :potential))

    if isempty(remainingvars)
        # they want to remove all variables except for prob column
        # uh ... 'singleton' table?
        return Table(DataFrame(potential = sum(f[!,:potential])))
    else
        # note that this will fail miserably if f is too large (~1E4 maybe?)
        #  nothing I can do; there is a github issue
        return Table(combine(df -> DataFrame(potential = sum(df[!,:potential])), DataFrames.groupby(f, remainingvars)))
    end
end

"""
Table normalization
Ensures that the `:potential` column sums to one
"""
function LinearAlgebra.normalize!(t::Table)
    t.potential[!,:potential] ./= sum(t.potential[!,:potential])

    return t
end

LinearAlgebra.normalize(t::Table) = normalize!(deepcopy(t))

"""
Given a Table, extract the rows which match the given assignment
"""
function Base.partialsort(t::Table, a::Assignment)
    f = t.potential

    commonNames = intersect(propertynames(f), keys(a))
    mask = trues(size(f, 1))
    for s in commonNames
        # mask &= (f[!,s] .== a[s])
        vals = (f[!,s] .== a[s])
        for (i,v) in enumerate(vals)
            mask[i] = mask[i] & v
        end
    end

    return Table(f[mask, :])
end

"""
takes a list of observations of assignments represented as a DataFrame
or a set of data samples (without :potential),
takes the unique assignments,
and estimates the associated probability of each assignment
based on its frequency of occurrence.
"""
function Distributions.fit(::Type{Table}, f::DataFrame)
    w = ones(size(f, 1))
    t = f
    if hasproperty(f, :potential)
        t = f[:, propertynames(t) .!= :potential]
        w = f[!,:potential]
    end
    # unique samples
    tu = unique(t)
    # add column with probabilities of unique samples
    tu[!,:potential] = Float64[sum(w[Bool[tu[j,:] == t[i,:] for i = 1:size(t,1)]]) for j = 1:size(tu,1)]
    tu[!,:potential] /= sum(tu[!,:potential])

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
#     if hasproperty(f, :potential)
#         w = f[!,:potential]
#     end
#
#     dfindex = find([hasproperty(a, n) for n in names(f)])
#     dfvalues = [a[n] for n in names(f)[!,dfindex]]'
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
