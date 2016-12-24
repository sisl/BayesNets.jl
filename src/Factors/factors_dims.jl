#
# Factors Reduce
#
# Dimension specific things, like broadcast, reduce, sum, etc.
# Functions (should) leave ft the same if dim ∉ ft

"""
    normalize(ft, dims, p=1)
    normalize(ft, p=1)

Normalize the factor so all instances of dims have (or the entire factors has)
p-norm of 1
"""
function LinAlg.normalize!(ft::Factor, dims; p::Int=1)
    if isa(dims, NodeName)
        dims = [dims]
    elseif isa(dims, Vector{NodeName})
        dims = unique(dims)
    else
        invalid_dims_error()
    end

    inds = indexin(dims, ft)
    inds = inds[inds .!= 0]

    if !isempty(inds)
        if p == 1
            total = sumabs(ft.probability, inds)
        elseif p == 2
            total = sumabs2(ft.probability, inds)
        else
            throw(ArgumentError("p = $(p) is not supported"))
        end

        ft.probability ./= total
    end

    return ft
end

function LinAlg.normalize!(ft::Factor; p::Int=1)
    if p == 1
        total = sumabs(ft.probability)
    elseif p == 2
        total = sumabs2(ft.probability)
    else
        throw(ArgumentError("p = $(p) is not supported"))
    end

    ft.probability ./= total

    return ft
end

"""
    reducedim(op, ft, dims, [v0])

Reduce dimensions `dims` in `ft` using function `op`.
See Base.reducedim for more details
"""
function Base.reducedim(op, ft::Factor,
        dims, v0=nothing)
    # opt for a more efficient version than reducedim!(deepcopy(ft))
    if isa(dims, NodeName)
        dims = [dims]
    elseif isa(dims, Vector{NodeName})
        dims = unique(dims)
    else
        invalid_dims_error()
    end

    inds = indexin(dims, ft)
    # get rid of dimensions not in ft
    inds = sort!(inds[inds .!= 0])

    if !isempty(inds)
        # needs to be a tuple for squeeze
        inds = (inds...)

        dims_new = deepcopy(ft.dimensions)
        deleteat!(dims_new, inds)

        if v0 != nothing
            p_new = squeeze(reducedim(op, ft.probability, inds, v0), inds)
        else
            p_new = squeeze(reducedim(op, ft.probability, inds), inds)
        end

        ft = Factor(dims_new, p_new)
    else
        ft = deepcopy(ft)
    end

    return ft
end

function reducedim!(op, ft::Factor, dims,
        v0=nothing)
    if isa(dims, NodeName)
        dims = [dims]
    elseif isa(dims, Vector{NodeName})
        dims = unique(dims)
    else
        invalid_dims_error()
    end

    inds = indexin(dims, ft)
    # get rid of dimensions not in ft
    inds = sort!(inds[inds .!= 0])

    if !isempty(dims)
        # needs to be a tuple for squeeze
        inds = (inds...)

        if v0 != nothing
            v_new = squeeze(reducedim(op, ft.probability, inds, v0), inds)
        else
            v_new = squeeze(reducedim(op, ft.probability, inds), inds)
        end

        ft.probability = v_new
        deleteat!(ft.dimensions, inds)
    end

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
Base.broadcast(f, ft::Factor, dims, values) =
    broadcast!(f, deepcopy(ft), dims, values)

"""
    broadcast!(f, ft, dims, values)

Broadcast a vector (or array of vectors) across the dimension(s) `dims`
Each vector in `values` will be broadcast acroos its respective dimension
in `dims`

See Base.broadcast for more info.
"""
function Base.broadcast!(f, ft::Factor, dims, values)
    if isa(dims, NodeName)
        dims = [dims]
        values = [values]
    elseif isa(dims, Vector{NodeName})
        if !allunique(dims)
            non_unique_dims_error()
        end
        if length(dims) != length(values)
            throw(ArgumentError("Number of dimensions does not " *
                        "match number of values to broadcast"))
        end
    else
        invalid_dims_error()
    end

    inds = indexin(dims, ft)
    # get rid of dimensions not in ft
    inds = inds[inds .!= 0]
    dims = dims[inds .!= 0]
    values = values[inds .!= 0]

    # check that either each vector matches the length of that dimension,
    # or that vector is a scalar
    if any( ( [size(ft, dims...)...] .!= map(length, values)) &
            (map(length, values) .!= 1) )
        throw(ArgumentError("Length of dimensions don't match " *
                    "lengths of broadcast array"))
    end

    # actually broadcast stuff
    for (i, val) in enumerate(values)
        if isa(val, Array)
            # reshape to the proper dimension, which needs a tuple
            reshape_dims = (vcat(ones(Int, inds[i]-1), length(val))...)
            val = reshape(val, reshape_dims)
        end
        broadcast!(f, ft.probability, ft.probability, val)
    end

    return ft
end

"""
Joins two factos. Only two kinds are allowed: inner and outer

Outer returns a Factor with the union of the two dimensions
The two factors are combined with Base.broadcast(op, ...)

Inner keeps only the intersect between the two
The extra dimensions are first reduced with reducedim(reducehow, ...)
and then the two factors are combined with:
    op(ft1[intersect].probability, ft2[intersect.probability])
"""
function Base.join(op, ft1::Factor, ft2::Factor, kind=:outer,
        reducehow=nothing, v0=nothing)
    # dimensions in common
    #  more than just names, so will be same type, states (hopefully?)
    common = intersect(ft1.dimensions, ft2.dimensions)

    if kind == :outer
        if length(ft1) < length(ft2)
            # avoid all the broadcast overhead with a larger array, maybe?
            ft2, ft1 = ft1, ft2
        end

        # the first dimensions are all from ft1
        new_dims = union(ft1.dimensions, ft2.dimensions)

        if ndims(ft2) != 0
            # permuate the common dimensions in ft2 to the front
            perm = collect(1:ndims(ft2))
            # find which dims in ft2 are shared
            is_common2 = indexin(ft2.dimensions, common) .!= 0
            # have their indices be moved to the front
            perm = vcat(perm[is_common2], perm[!is_common2])
            temp = permutedims(ft2.probability, perm)

            # now reshape it by lining up the common dims in ft2 with those in ft1
            #  boolean for which new dimensions come from ft1 only
            is_unique1 = indexin(new_dims, setdiff(ft1.dimensions, common))
            # set those dims to have dimension 1 for data in ft2
            reshape_lengths = map(length, new_dims)
            nd1 = ndims(ft1)
            new_v = duplicate(ft1.probability, (reshape_lengths[(nd1+1):end]...))
            reshape_lengths[is_unique1 .!= 0] = 1
            temp = reshape(temp, (reshape_lengths...))
        else
            new_v = copy(ft1.probability)
            temp = ft2.probability
        end

        # ft1 has at at least as many elements as ft2, ft1 == 0 -> ft2 == 0
        if ndims(ft1) == 0
            new_v = squeeze([op(new_v[1], temp[1])], 1)
        else
            # broadcast will automatically expand the later dimensions of ft1.probability
            broadcast!(op, new_v, new_v, temp)
        end
    elseif kind == :inner
        error("inner is still unsupported")

        new_dims = getdim(ft1, common)

        if isempty(common)
            # weird magic for a zero-dimensional array
            v_new = squeeze(zero(eltype(ft1), 0), 1)
        else
            if reducehow == nothing
                throw(ArgumentError("Need reducehow"))
            end

            inds1 = (findin(ft1.dimensions, common)...)
            inds2 = (findin(ft2.dimensions, common)...)

            if v0 != nothing
                v1_new = squeeze(reducedim(op, ft1.probability, inds1, v0), inds)
                v2_new = squeeze(reducedim(op, ft2.probability, inds2, v0), inds)
            else
                v1_new = squeeze(reducedim(op, ft1.probability, inds1), inds)
                v2_new = squeeze(reducedim(op, ft2.probability, inds2), inds)
        new_m = zeros(eltype(ft1.probability), reshape_lengths)
            end
            v_new = op(v1_new, v2_new)
        end
    else
        throw(ArgumentError("$(kind) is not a supported join type"))
    end

    return Factor(new_dims, new_v)
end

*(ft1::Factor, ft2::Factor) = join(*, ft1, ft2)
+(ft1::Factor, ft2::Factor) = join(+, ft1, ft2)

