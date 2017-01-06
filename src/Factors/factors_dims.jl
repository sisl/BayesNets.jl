#
# Factors Reduce
#
# Dimension specific things, like broadcast, reduce, sum, etc.
# Functions (should) leave ft the same if dim ∉ ft


"""
    normalize!(ft, dims; p=1)
    normalize!(ft; p=1)

Return a normalized copy of the factor so all instances of dims have
(or the entire factors has) p-norm of 1
"""
LinAlg.normalize(ft::Factor, x...; k...) = normalize!(deepcopy(ft), x...; k...)

"""
    normalize!(ft, dims; p=1)
    normalize!(ft; p=1)

Normalize the factor so all instances of dims have (or the entire factors has)
p-norm of 1
"""
function LinAlg.normalize!(ft::Factor, dims::NodeNames; p::Int=1)
    dims = unique(_ckdimtype(dims))
    _ckdimvalid(dims, ft)

    inds = indexin(dims, ft)

    if !isempty(inds)
        if p == 1
            total = sumabs(ft.potential, inds)
        elseif p == 2
            total = sumabs2(ft.potential, inds)
        else
            throw(ArgumentError("p = $(p) is not supported"))
        end

        ft.potential ./= total
    end

    return ft
end

function LinAlg.normalize!(ft::Factor; p::Int=1)
    if p == 1
        total = sumabs(ft.potential)
    elseif p == 2
        total = sumabs2(ft.potential)
    else
        throw(ArgumentError("p = $(p) is not supported"))
    end

    ft.potential ./= total

    return ft
end

# reduce the dimension and then squeeze them out
_reddim(op, ft::Factor, inds::Tuple, ::Void) =
            squeeze(reducedim(op, ft.potential, inds), inds)
_reddim(op, ft::Factor, inds::Tuple, v0) =
            squeeze(reducedim(op, ft.potential, inds, v0), inds)

"""
    reducedim(op, ft, dims, [v0])

Reduce dimensions `dims` in `ft` using function `op`.
See Base.reducedim for more details
"""
function Base.reducedim(op, ft::Factor,
        dims::NodeNames, v0=nothing)
    # a (possibly?) more efficient version than reducedim!(deepcopy(ft))
    dims = _ckdimtype(dims)
    _ckdimvalid(dims, ft)

    # needs to be a tuple for squeeze
    inds = (indexin(dims, ft)...)

    dims_new = deepcopy(ft.dimensions)
    deleteat!(dims_new, inds)

    v_new = _reddim(op, ft, inds, v0)
    ft = Factor(dims_new, v_new)

    return ft
end

function reducedim!(op, ft::Factor, dims::NodeNames,
        v0=nothing)
    dims = _ckdimtype(dims)
    _ckdimvalid(dims, ft)

    # needs to be a tuple for squeeze
    inds = (indexin(dims, ft)...)

    deleteat!(ft.dimensions, inds)
    ft.potential = _reddim(op, ft, inds, v0)

    return ft
end

Base.sum(ft::Factor, dims) = reducedim(+, ft, dims)
Base.sum!(ft::Factor, dims) = reducedim!(+, ft, dims)
Base.prod(ft::Factor, dims) = reducedim(*, ft, dims)
Base.prod!(ft::Factor, dims) = reducedim!(*, ft, dims)
Base.maximum(ft::Factor, dims) = reducedim(max, ft, dims)
Base.maximum!(ft::Factor, dims) = reducedim!(max, ft, dims)
Base.minimum(ft::Factor, dims) = reducedim(min, ft, dims)
Base.minimum!(ft::Factor, dims) = reducedim!(min, ft, dims)

"""
    broadcast(f, ft, dims, values)

Broadcast a vector (or array of vectors) across the dimension(s) `dims`
Each vector in `values` will be broadcast acroos its respective dimension
in `dims`

See Base.broadcast for more info.
"""
Base.broadcast(f, ft::Factor, dims::NodeNames, values) =
    broadcast!(f, deepcopy(ft), dims, values)

"""
    broadcast!(f, ft, dims, values)

Broadcast a vector (or array of vectors) across the dimension(s) `dims`
Each vector in `values` will be broadcast acroos its respective dimension
in `dims`

See Base.broadcast for more info.
"""
function Base.broadcast!(f, ft::Factor, dims::NodeNames, values)
    if isa(dims, NodeName)
        dims = [dims]
        values = [values]
    end

    _ckdimunq(dims)
    _ckdimvalid(dims, ft)

    (length(dims) != length(values)) &&
        throw(ArgumentError("Number of dimensions does not " *
                    "match number of values to broadcast"))

    # broadcast will check if the dimensions of each value are valid

    inds = indexin(dims, ft)

    reshape_dims = ones(Int, ndims(ft))
    new_values = Vector{Array{Float64}}(length(values))

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

    broadcast!(f, ft.potential, ft.potential, new_values...)

    return ft
end

"""
    join(op, ft1, ft2, :outer, [v0])
    join(op, ft1, ft2, :inner, [reducehow], [v0])

Performs either an inner or outer join,

An outer join returns a Factor with the union of the two dimensions
The two factors are combined with Base.broadcast(op, ...)

An inner join keeps the dimensions in common between the two Factors.
The extra dimensions are reduced with 
    reducedim(reducehow, ...)
and then the two factors are combined with:
    op(ft1[common_dims].potential, ft2[common_dims].potential)
"""
function Base.join(op, ft1::Factor, ft2::Factor, kind::Symbol=:outer,
        reducehow=nothing, v0=nothing)
    # avoid all the broadcast overhead with a larger array (ideally)
    # more useful for edge cases where one (or both) is (are) singleton
    if length(ft1) < length(ft2)
        ft2, ft1 = ft1, ft2
    end

    common = intersect(ft1.dimensions, ft2.dimensions)
    index_common1 = indexin(common, ft1.dimensions)
    index_common2 = indexin(common, ft2.dimensions)

    if [size(ft1)[index_common1]...] != [size(ft2)[index_common2]...]
        throw(DimensionMismatch("Common dimensions must have same size"))
    end

    if kind == :outer
        # the first dimensions are all from ft1
        new_dims = union(ft1.dimensions, ft2.dimensions)

        if ndims(ft2) != 0
            # permute the common dimensions in ft2 to the beginning,
            #  in the order that they appear in ft1 (and therefore new_dims)
            unique1 = setdiff(ft1.dimensions, common)
            unique2 = setdiff(ft2.dimensions, common)
            # these will also be the same indices for new_dims
            index_unique1 = indexin(unique1, ft1.dimensions)
            index_unique2 = indexin(unique2, ft2.dimensions)
            perm = vcat(index_common2, index_unique2)
            temp = permutedims(ft2.potential, perm)

            # reshape by lining up the common dims in ft2 with those in ft1
            size_unique2 = size(ft2)[index_unique2]
            # set those dims to have dimension 1 for data in ft2
            reshape_lengths = vcat(size(ft1)..., size_unique2...)
            #new_v = duplicate(ft1.potential, size_unique2)
            new_v = Array{Float64}(reshape_lengths...)
            reshape_lengths[index_unique1] = 1
            temp = reshape(temp, (reshape_lengths...))
        else
            new_v = similar(ft1.potential)
            temp = ft2.potential
        end

        # ndims(ft1) == 0 implies ndims(ft2) == 0
        if ndims(ft1) == 0
            new_v = squeeze([op(ft1.potential[1], temp[1])], 1)
        else
            broadcast!(op, new_v, ft1.potential, temp)
        end
    elseif kind == :inner
        error("Inner joins are (still) umimplemented")

        new_dims = getdim(ft1, common)

        if isempty(common)
            # weird magic for a zero-dimensional array
            v_new = squeeze(zero(eltype(ft1), 0), 1)
        else
            if reducehow == nothing
                throw(ArgumentError("`reducehow` is needed to reduce " *
                            "non-common dimensions"))
            end

            inds1 = (findin(ft1.dimensions, common)...)
            inds2 = (findin(ft2.dimensions, common)...)

            if v0 != nothing
                v1_new = squeeze(reducedim(op, ft1.potential, inds1, v0), inds)
                v2_new = squeeze(reducedim(op, ft2.potential, inds2, v0), inds)
            else
                v1_new = squeeze(reducedim(op, ft1.potential, inds1), inds)
                v2_new = squeeze(reducedim(op, ft2.potential, inds2), inds)
            end

            v_new = op(v1_new, v2_new)
        end
    else
        throw(ArgumentError("$(kind) is not a supported join type"))
    end

    return Factor(new_dims, new_v)
end

*(ft1::Factor, ft2::Factor) = join(*, ft1, ft2)
/(ft1::Factor, ft2::Factor) = join(/, ft1, ft2)
+(ft1::Factor, ft2::Factor) = join(+, ft1, ft2)
-(ft1::Factor, ft2::Factor) = join(-, ft1, ft2)

