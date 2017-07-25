#
# Main Factor Code
#

# THE MOST BASIC ASSUMPTION IS THAT ALL VARIABLES ARE CATEGORICAL AND THEREFORE
# Base.OneTo WORTHY. IF THAT IS VIOLATED, NOTHING WILL WORK

"""
    Factor(dims, potential)

Create a Factor corresponding to the potential.
"""
mutable struct Factor
    dimensions::NodeNames
    potential::Array{Float64} # Unnormalized probability
                              # In most cases this will be a probability

    function Factor(dims::NodeNameUnion, potential::Array{Float64})
        dims = convert(NodeNames, dims)
        _ckeck_dims_unique(dims)

        (length(dims) != ndims(potential)) &&
            throw(DimensionMismatch("`potential` must have as many " *
                        "dimensions as dims"))

        (:potential in dims) &&
            warn("Having a dimension called `potential` will cause problems")

        return new(dims, potential)
    end
end

"""
    Factor(dims, lengths, fill_value=0)

Create a factor with dimensions `dims`, each with lengths corresponding to
`lengths`. `fill_value` will fill the potential array with that value.
To keep uninitialized, use `fill_value=nothing`.
"""
Factor(dims::NodeNames, lengths::Vector{Int}, ::Void) =
    Factor(dims, Array{Float64}(lengths...))

Factor(dims::NodeNames, lengths::Vector{Int}, fill_value::Number=0) =
    Factor(dims, fill(Float64(fill_value), lengths...))

Factor(dim::NodeName, length::Int, ::Void) =
    Factor([dims], Array{Float64}(length))

Factor(dim::NodeName, length::Int, fill_value::Number=0) =
    Factor(dims, fill(Float64(fill_value), length))

"""
    convert(DiscreteCPD, cpd)

Construct a Factor from a DiscreteCPD.
"""
function Base.convert(::Type{Factor}, cpd::DiscreteCPD)
    dims = vcat(name(cpd), parents(cpd))
    lengths = tuple(ncategories(cpd), cpd.parental_ncategories...)
    p = Array{Float64}(lengths)
    p[:] = vcat([d.p for d in cpd.distributions]...)
    return Factor(dims, p)
end
Base.mimewritable(::MIME"text/html", ϕ::DiscreteCPD) = true
Base.show(io::IO, cpd::DiscreteCPD) = show(io, convert(Factor, cpd))
Base.show(io::IO, a::MIME"text/html", cpd::DiscreteCPD) = show(io, a, convert(DataFrame, convert(Factor, cpd)))


"""
    Factor(bn, name, evidence=Assignment())

Create a factor for a node, given some evidence.
"""
function Factor(bn::DiscreteBayesNet, name::NodeName, evidence::Assignment=Assignment())
    cpd = get(bn, name)
    ϕ = convert(Factor, cpd)
    return ϕ[evidence]
end

###############################################################################
#                   Methods

"""
    similar(ϕ)

Return a factor similar to `ϕ` with unitialized values
"""
Base.similar(ϕ::Factor) = Factor(ϕ.dimensions, similar(ϕ.potential))

"""
Returns Float64
"""
Base.eltype(ϕ::Factor) = Float64

"""
Names of each dimension
"""
Base.names(ϕ::Factor) = ϕ.dimensions

Base.ndims(ϕ::Factor) = ndims(ϕ.potential)

"""
    size(ϕ, [dims...])

Returns a tuple of the dimensions of `ϕ`
"""
Base.size(ϕ::Factor) = size(ϕ.potential)
Base.size(ϕ::Factor, dim::NodeName) = size(ϕ.potential, indexin(dim, ϕ))
Base.size{N}(ϕ::Factor, dims::Vararg{NodeName, N}) =
    ntuple(k -> size(ϕ, dims[k]), Val{N})

"""
Total number of elements in Factor (potential)
"""
Base.length(ϕ::Factor) = length(ϕ.potential)

"""
    in(dim, ϕ) -> Bool

Return true if `dim` is in the Factor `ϕ`
"""
Base.in(dim::NodeName, ϕ::Factor) = dim in names(ϕ)

"""
    indexin(dims, ϕ)

Return the index of dimension `dim` in `ϕ`, or 0 if not in `ϕ`.
"""
Base.indexin(dim::NodeName, ϕ::Factor) = findnext(ϕ.dimensions, dim, 1)
Base.indexin(dims::NodeNames, ϕ::Factor) = indexin(dims, names(ϕ))


"""
    rand!(ϕ)

Fill with random values
"""
Base.rand!(ϕ::Factor) = rand!(ϕ.potential)

"""
Appends a new dimension to a Factor
"""
@inline function Base.push!(ϕ::Factor, dim::NodeName, length::Int)
    if dim in names(ϕ)
        error("Dimension $(dim) already exists")
    end

    p = duplicate(ϕ.potential, (length, ))
    ϕ.dimensions = push!(ϕ.dimensions, dim)
    ϕ.potential = p

    return ϕ
end

@inline function Base.permutedims!(ϕ::Factor, perm)
    ϕ.potential = permutedims(ϕ.potential, perm)
    ϕ.dimensions = ϕ.dimensions[perm]
    return ϕ
end

Base.permutedims(ϕ::Factor, perm) = permutedims!(deepcopy(ϕ), perm)

"""
    pattern(ϕ, [dims])

Return an array with the pattern of each dimension's state for all possible
instances
"""
function pattern(ϕ::Factor, dims)
    inds = indexin(dims, ϕ)

    zero_loc = findfirst(inds, 0)
    zero_loc == 0 || not_in_factor_error(dims[zero_loc])

    lens = [size(ϕ)...]

    inners = vcat(1, cumprod(lens[1:(end-1)]))
    outers = Int[(length(ϕ) ./ lens[inds] ./ inners[inds])...]

    hcat([repeat(collect(1:l), inner=i, outer=o) for (l, i, o) in
        zip(lens[inds], inners[inds], outers)]...)
end

function pattern(ϕ::Factor)
    lens = [size(ϕ)...]

    inners = vcat(1, cumprod(lens[1:(end-1)]))
    outers = Int[(length(ϕ) ./ lens ./ inners)...]

    hcat([repeat(collect(1:l), inner=i, outer=o) for (l, i, o) in
            zip(lens, inners, outers)]...)
end

