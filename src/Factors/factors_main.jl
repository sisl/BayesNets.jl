#
# Main Factor Code
#

# THE MOST BASIC ASSUMPTION IS THAT ALL VARIABLES ARE CATEGORICAL AND THEREFORE
# Base.OneTo WORTHY. IF THAT IS VIOLATED, NOTHING WILL WORK

type Factor
    dimensions::Vector{NodeName}
    probability::Array{Float64}

    function Factor(dims::Vector{NodeName}, probability::Array{Float64})
        if length(dims) != ndims(probability)
            throw(DimensionMismatch("`probability` must have as many " *
                        "dimensions as dims"))
        end

        if !allunique(dims)
            non_unique_dims_error()
        end

        if :probability in dims
            warn("Having a dimension called `probability` will cause problems")
        end

        return new(dims, probability)
    end
end

"""
    Factor(dims, lengths, fill=0)

Create a factor with dimensions `dims`, each with lengths corresponding to
`lengths`. `fill` will fill the probability array with that value.
To keep uninitialized, use nothing.
"""
function Factor(dims::Vector{NodeName}, lengths::Vector{Int}, fill_value=0)
    if fill_value == nothing
        p = Array{Float64}(lengths...)
    else
        p = fill(Float64(fill_value), lengths...)
    end

    return Factor(dims, p)
end

"""
    Factor(bn, name, evidence::Assignment())

Create a factor for a node, given some evidence.
"""
function Factor(bn::DiscreteBayesNet, name::NodeName,
        evidence::Assignment=Assignment())
    cpd = get(bn, name)
    dims = vcat(name, parents(bn, name))
    lengths = ntuple(i -> ncategories(bn, dims[i]), length(dims))

    p = Array{Float64}(lengths)
    p[:] = vcat([d.p for d in cpd.distributions]...)
    ft = Factor(dims, p)

    return ft[evidence]
end

###############################################################################
#                   Methods

"""
    similar(ft)

Return a factor similar to `ft` with unitialized values
"""
Base.similar(ft::Factor) = Factor(ft.dimensions, similar(ft.probability))

"""
Returns Float64
"""
Base.eltype(ft::Factor) = Float64

"""
Names of each dimension
"""
Base.names(ft::Factor) = ft.dimensions

Base.ndims(ft::Factor) = ndims(ft.probability)

"""
    size(ft, [dims...])

Returns a tuple of the dimensions of `ft`
"""
Base.size(ft::Factor) = size(ft.probability)
Base.size(ft::Factor, dim::NodeName) = size(ft.probability, indexin(dim, ft))
Base.size{N}(ft::Factor, dims::Vararg{NodeName, N}) =
    ntuple(k -> size(ft, dims[k]), Val{N})

"""
Total number of elements total
"""
Base.length(ft::Factor) = length(ft.probability)

"""
    in(dim, ft) -> Bool

Return true if `dim` is in the Factor `ft`
"""
Base.in(dim::NodeName, ft::Factor) = dim in names(ft)

"""
    indexin(dims, ft)

Return the index of dimension `dim` in `ft`, or 0 if not in `ft`.
"""
Base.indexin(dim::NodeName, ft::Factor) = findnext(ft.dimensions, dim, 1)
Base.indexin(dims::Vector{NodeName}, ft::Factor) = indexin(dims, names(ft))


"""
    rand!(ft)

Fill with random values
"""
Base.rand!(ft) = rand!(ft.probability)

"""
Appends a new dimension to a Factor
"""
@inline function Base.push!(ft::Factor, dim::NodeName, length::Int)
    if dim in names(ft)
        error("Dimension $(dim) already exists")
    end

    p = duplicate(ft.probability, (length, ))
    ft.dimensions = push!(ft.dimensions, dim)
    ft.probability = p

    return ft
end

@inline function Base.permutedims!(ft::Factor, perm)
    ft.probability = permutedims(ft.probability, perm)
    ft.dimensions = ft.dimensions[perm]
    return ft
end

Base.permutedims(ft::Factor, perm) = permutedims!(deepcopy(ft), perm)

"""
    pattern(ft, [dims])

Return an array with the pattern of each dimension's state for all possible
instances
"""
function pattern(ft::Factor, dims)
    inds = indexin(dims, ft)

    zero_loc = findfirst(inds, 0)
    zero_loc == 0 || not_in_factor_error(dims[zero_loc])

    lens = [size(ft)...]

    inners = vcat(1, cumprod(lens[1:(end-1)]))
    outers = Int[(length(ft) ./ lens[inds] ./ inners[inds])...]

    hcat([repeat(collect(1:l), inner=i, outer=o) for (l, i, o) in
        zip(lens[inds], inners[inds], outers)]...)
end

function pattern(ft::Factor)
    lens = [size(ft)...]

    inners = vcat(1, cumprod(lens[1:(end-1)]))
    outers = Int[(length(ft) ./ lens ./ inners)...]

    hcat([repeat(collect(1:l), inner=i, outer=o) for (l, i, o) in
            zip(lens, inners, outers)]...)
end

