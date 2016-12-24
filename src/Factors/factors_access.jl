#
# Factors Access
#
# For ft[:A] syntax and such

# Column access (ft[:A]) returns the dimension for comparisons and stuff
Base.getindex(ft::Factor, dim::NodeName) = pattern(ft, dim)
Base.getindex(ft::Factor, dims::Vector{NodeName}) = pattern(ft, dims)

# Index by number gets that ind
Base.getindex(ft::Factor, i::Int) = ft.probability[i]
Base.getindex(ft::Factor, I::Vararg{Int}) = ft.probability[I]

"""
    getindex(ft, a)

Get values with dimensions consistent with an assignment.
Colons select entire dimension
"""
function Base.getindex(ft::Factor, a::Assignment)
    # more complicated logic than finding indices b/c also updating
    #  dimensions
    inds = Array{Any}(ndims(ft))
    inds[:] = Colon()

    new_dims = deepcopy(ft. dimensions)
    keep = trues(length(new_dims))

    for (i, dim) in enumerate(ft.dimensions)
        if haskey(a, dim.name)
            val = a[dim.name]

            if isa(val, Colon)
                # nothing to be done, really
                continue
            end

            if isa(val, BitArray)
                length(dim) == length(val) || throw(ArgumentError("Length " *
                            "of BitArray does not match dimension $(dim.name)"))
                ind = val
            else
                # index in each dimension is location of value
                ind = indexin(val, dim)

                # also works for scalars (hopefully)
                zero_loc = findfirst(ind .== 0)
                if zero_loc != 0
                    throw(ArgumentError("$(val[zero_loc]) is not a valid" *
                                " state of $(name(dim))"))
                end
            end

            if length(ind) == 1
                # singleton access, drop the dimension
                keep[i] = false
                # make sure the index is a scalar so array access drops
                #  that dimension
                ind = first(ind)
            else
                # update the dimension
                @inbounds new_dims[i] = update(dim, ind)
            end

            inds[i] = ind
        end
    end

    new_dims = new_dims[keep]
    new_v = ft.probability[inds...]

    # in case array access returns a scalar and not an array
    if ndims(new_v) == 0
        new_v = squeeze([new_v], 1)
    end

    return Factor(new_dims, new_v)
end

function Base.setindex!(ft::Factor, v, a::Assignment)
    return ft.probability[_translate_index(ft, a)...] = v
end

function _translate_index(ft::Factor, a::Assignment)
    ind = Array{Any}(ndims(ft))
    # all dimensions are accessed by default
    ind[:] = Colon()

    for (i, dim) in enumerate(ft.dimensions)
        if haskey(a, dim.name)
            val = a[dim.name]

            if isa(val, BitArray)
                length(dim) == length(val) || throw(ArgumentError("Length " *
                            "of BitArray does not match dimension $(dim.name)"))
                ind[i] = val
            else
                # index in each dimension is location of value
                ind[i] = indexin(val, dim)

                if any(ind[i] .== 0)
                    # if assignment[d] is not valid,shortcircuit and
                    #  return an empty array
                    return []
                end
            end
        end
    end

    return ind
end


Base.sub2ind(ft::Factor, i...) = sub2ind(size(ft), i...)
Base.ind2sub(ft::Factor, i) = ind2sub(size(ft), i)

