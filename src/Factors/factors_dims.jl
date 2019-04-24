#
# Factors Reduce
#
# Dimension specific things, like broadcast, reduce, sum, etc.
# Functions (should) leave ϕ the same if dim ∉ ϕ


"""
    normalize!(ϕ, dims; p=1)
    normalize!(ϕ; p=1)

Return a normalized copy of the factor so all instances of dims have
(or the entire factors has) p-norm of 1
"""
LinearAlgebra.normalize(ϕ::Factor, x...; k...) = normalize!(deepcopy(ϕ), x...; k...)

"""
    normalize!(ϕ, dims; p=1)
    normalize!(ϕ; p=1)

Normalize the factor so all instances of dims have (or the entire factors has)
p-norm of 1
"""
function LinearAlgebra.normalize!(ϕ::Factor, dims::NodeNameUnion; p::Int=1)
    dims = unique(convert(NodeNames, dims))
    _check_dims_valid(dims, ϕ)

    inds = indexin(dims, ϕ)

    if !isempty(inds)
        if p == 1
            total = sum(abs, ϕ.potential, inds)
        elseif p == 2
            total = sum(abs2, ϕ.potential, inds)
        else
            throw(ArgumentError("p = $(p) is not supported"))
        end

        ϕ.potential ./= total
    end

    return ϕ
end

function LinearAlgebra.normalize!(ϕ::Factor; p::Int=1)
    if p == 1
        total = sum(abs, ϕ.potential)
    elseif p == 2
        total = sum(abs2, ϕ.potential)
    else
        throw(ArgumentError("p = $(p) is not supported"))
    end

    ϕ.potential ./= total

    return ϕ
end

# reduce the dimension and then squeeze them out
_reddim(op, ϕ::Factor, inds::Tuple, ::Nothing) =
            dropdims(reduce(op, ϕ.potential, dims=inds), dims=inds)
_reddim(op, ϕ::Factor, inds::Tuple, v0) =
            dropdims(reducedim(op, ϕ.potential, inds, v0), dims=inds)

"""
    reducedim(op, ϕ, dims, [v0])

Reduce dimensions `dims` in `ϕ` using function `op`.
"""
function reducedim(op, ϕ::Factor, dims::NodeNameUnion, v0=nothing)
    # a (possibly?) more efficient version than reducedim!(deepcopy(ϕ))
    dims = convert(NodeNames, dims)
    _check_dims_valid(dims, ϕ)

    # needs to be a tuple for squeeze
    inds = (indexin(dims, ϕ)...,)

    dims_new = deepcopy(ϕ.dimensions)
    deleteat!(dims_new, inds)

    v_new = _reddim(op, ϕ, inds, v0)
    ϕ = Factor(dims_new, v_new)

    return ϕ
end

function reducedim!(op, ϕ::Factor, dims::NodeNameUnion, v0=nothing)
    dims = convert(NodeNames, dims)
    _check_dims_valid(dims, ϕ)

    # needs to be a tuple for squeeze
    inds = (indexin(dims, ϕ)...,)

    deleteat!(ϕ.dimensions, inds)
    ϕ.potential = _reddim(op, ϕ, inds, v0)

    return ϕ
end

Base.sum(ϕ::Factor, dims::NodeNameUnion) = reducedim(+, ϕ, dims)
Base.sum!(ϕ::Factor, dims::NodeNameUnion) = reducedim!(+, ϕ, dims)
Base.prod(ϕ::Factor, dims::NodeNameUnion) = reducedim(*, ϕ, dims)
Base.prod!(ϕ::Factor, dims::NodeNameUnion) = reducedim!(*, ϕ, dims)
Base.maximum(ϕ::Factor, dims::NodeNameUnion) = reducedim(max, ϕ, dims)
Base.maximum!(ϕ::Factor, dims::NodeNameUnion) = reducedim!(max, ϕ, dims)
Base.minimum(ϕ::Factor, dims::NodeNameUnion) = reducedim(min, ϕ, dims)
Base.minimum!(ϕ::Factor, dims::NodeNameUnion) = reducedim!(min, ϕ, dims)

"""
    broadcast(f, ϕ, dims, values)

Broadcast a vector (or array of vectors) across the dimension(s) `dims`
Each vector in `values` will be broadcast acroos its respective dimension
in `dims`

See Base.broadcast for more info.
"""
Base.broadcast(f, ϕ::Factor, dims::NodeNameUnion, values) =
    broadcast!(f, deepcopy(ϕ), dims, values)

"""
    broadcast!(f, ϕ, dims, values)

Broadcast a vector (or array of vectors) across the dimension(s) `dims`
Each vector in `values` will be broadcast acroos its respective dimension
in `dims`

See Base.broadcast for more info.
"""
function Base.broadcast!(f, ϕ::Factor, dims::NodeNameUnion, values)
    if isa(dims, NodeName)
        dims = [dims]
        values = [values]
    end

    _ckeck_dims_unique(dims)
    _check_dims_valid(dims, ϕ)

    (length(dims) != length(values)) &&
        throw(ArgumentError("Number of dimensions does not " *
                    "match number of values to broadcast"))

    # broadcast will check if the dimensions of each value are valid

    inds = indexin(dims, ϕ)

    reshape_dims = ones(Int, ndims(ϕ))
    new_values = Vector{Array{Float64}}(undef, length(values))

    for (i, val) in enumerate(values)
        if isa(val, Vector{Float64})
            # reshape to the proper dimension
            dim_loc = inds[i]
            @inbounds reshape_dims[dim_loc] = length(val)
            new_values[i] = reshape(val, reshape_dims...)
            @inbounds reshape_dims[dim_loc] = 1
        elseif isa(val, Float64)
            new_values[i] = [val]
        else
            throw(TypeError(:broadcast!, "Invalid broadcast value",
                        Union{Float64, Vector{Float64}}, val))
        end
    end

    broadcast!(f, ϕ.potential, ϕ.potential, new_values...)

    return ϕ
end

"""
    join(op, ϕ1, ϕ2, :outer, [v0])
    join(op, ϕ1, ϕ2, :inner, [reducehow], [v0])

Performs either an inner or outer join,

An outer join returns a Factor with the union of the two dimensions
The two factors are combined with Base.broadcast(op, ...)

An inner join keeps the dimensions in common between the two Factors.
The extra dimensions are reduced with
    reducedim(reducehow, ...)
and then the two factors are combined with:
    op(ϕ1[common_dims].potential, ϕ2[common_dims].potential)
"""
function Base.join(op, ϕ1::Factor, ϕ2::Factor, kind::Symbol=:outer,
        reducehow=nothing, v0=nothing)
    # avoid all the broadcast overhead with a larger array (ideally)
    # more useful for edge cases where one (or both) is (are) singleton
    if length(ϕ1) < length(ϕ2)
        ϕ2, ϕ1 = ϕ1, ϕ2
    end

    common = intersect(ϕ1.dimensions, ϕ2.dimensions)
    index_common1 = indexin(common, ϕ1.dimensions)
    index_common2 = indexin(common, ϕ2.dimensions)

    if [size(ϕ1)[index_common1]...] != [size(ϕ2)[index_common2]...]
        throw(DimensionMismatch("Common dimensions must have same size"))
    end

    if kind == :outer
        # the first dimensions are all from ϕ1
        new_dims = union(ϕ1.dimensions, ϕ2.dimensions)

        if ndims(ϕ2) != 0
            # permute the common dimensions in ϕ2 to the beginning,
            #  in the order that they appear in ϕ1 (and therefore new_dims)
            unique1 = setdiff(ϕ1.dimensions, common)
            unique2 = setdiff(ϕ2.dimensions, common)
            # these will also be the same indices for new_dims
            index_unique1 = indexin(unique1, ϕ1.dimensions)
            index_unique2 = indexin(unique2, ϕ2.dimensions)
            perm = vcat(index_common2, index_unique2)
            temp = permutedims(ϕ2.potential, perm)

            # reshape by lining up the common dims in ϕ2 with those in ϕ1
            size_unique2 = size(ϕ2)[index_unique2]
            # set those dims to have dimension 1 for data in ϕ2
            reshape_lengths = vcat(size(ϕ1)..., size_unique2...)
            #new_v = duplicate(ϕ1.potential, size_unique2)
            new_v = Array{Float64}(undef, reshape_lengths...)
            reshape_lengths[index_unique1] .= 1
            temp = reshape(temp, (reshape_lengths...,))
        else
            new_v = similar(ϕ1.potential)
            temp = ϕ2.potential
        end

        # ndims(ϕ1) == 0 implies ndims(ϕ2) == 0
        if ndims(ϕ1) == 0
            new_v = dropdims([op(ϕ1.potential[1], temp[1])], dims=1)
        else
            broadcast!(op, new_v, ϕ1.potential, temp)
        end
    elseif kind == :inner
        error("Inner joins are (still) umimplemented")

        new_dims = getdim(ϕ1, common)

        if isempty(common)
            # weird magic for a zero-dimensional array
            v_new = squeeze(zero(eltype(ϕ1), 0), 1)
        else
            if reducehow == nothing
                throw(ArgumentError("`reducehow` is needed to reduce " *
                            "non-common dimensions"))
            end

            inds1 = (findin(ϕ1.dimensions, common)...,)
            inds2 = (findin(ϕ2.dimensions, common)...,)

            if v0 != nothing
                v1_new = squeeze(reducedim(op, ϕ1.potential, inds1, v0), inds)
                v2_new = squeeze(reducedim(op, ϕ2.potential, inds2, v0), inds)
            else
                v1_new = squeeze(reducedim(op, ϕ1.potential, inds1), inds)
                v2_new = squeeze(reducedim(op, ϕ2.potential, inds2), inds)
            end

            v_new = op(v1_new, v2_new)
        end
    else
        throw(ArgumentError("$(kind) is not a supported join type"))
    end

    return Factor(new_dims, new_v)
end

*(ϕ1::Factor, ϕ2::Factor) = join(*, ϕ1, ϕ2)
/(ϕ1::Factor, ϕ2::Factor) = join(/, ϕ1, ϕ2)
+(ϕ1::Factor, ϕ2::Factor) = join(+, ϕ1, ϕ2)
-(ϕ1::Factor, ϕ2::Factor) = join(-, ϕ1, ϕ2)

