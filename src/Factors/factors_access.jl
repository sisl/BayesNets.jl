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
Colons select entire dimension.
"""
function Base.getindex(ft::Factor, a::Assignment)
    # more complicated logic than finding indices b/c also updating
    #  dimensions
    inds = _translate_index(ft, a)
    keep = inds .== Colon()
    new_dims = ft.dimensions[keep]
    @inbounds new_p = ft.probability[inds...]

    # in case array access returns a scalar and not an array
    # as always, the weird edge case to get a zero-dimensional array and not
    #  a scalar
    if ndims(new_p) == 0
        new_p = squeeze([new_p], 1)
    end

    return Factor(new_dims, new_p)
end

function Base.setindex!(ft::Factor, v, a::Assignment)
    @inbounds return ft.probability[_translate_index(ft, a)...] = v
end

@inline function _translate_index(ft::Factor, a::Assignment)
    inds = Array{Any}(ndims(ft))
    inds[:] = Colon()

    for (i, dim) in enumerate(ft.dimensions)
        if haskey(a, dim)
            ind = a[dim]

            if isa(ind, Colon)
                continue
            elseif isa(ind, Int)
                if ind < 1 || ind > size(ft, dim)
                    throw(BoundsError(dim, ind))
                end
            else
                throw(TypeError(:getindex, "Invalid state for dimension $dim",
                            Int, typeof(ind)))
            end

            inds[i] = ind
        end
    end

    return inds
end

Base.sub2ind(ft::Factor, i...) = sub2ind(size(ft), i...)
Base.ind2sub(ft::Factor, i) = ind2sub(size(ft), i)

