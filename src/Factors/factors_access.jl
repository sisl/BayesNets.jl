#
# Factors Access
#
# For ϕ[:A] syntax and such

# Column access (ϕ[:A]) returns the dimension for comparisons and stuff
Base.getindex(ϕ::Factor, dims::NodeNameUnion) = pattern(ϕ, dims)

# Index by number gets that ind
Base.getindex(ϕ::Factor, i::Int) = ϕ.potential[i]
Base.getindex(ϕ::Factor, I::Vararg{Int}) = ϕ.potential[I]

"""
    getindex(ϕ, a)

Get values with dimensions consistent with an assignment.
Colons select entire dimension.
"""
function Base.getindex(ϕ::Factor, a::Assignment)
    # more complicated logic than finding indices b/c also updating
    #  dimensions
    inds = _translate_index(ϕ, a)
    keep = inds .== Colon()
    new_dims = ϕ.dimensions[keep]
    @inbounds new_p = ϕ.potential[inds...]

    # in case array access returns a scalar and not an array
    # as always, the weird edge case to get a zero-dimensional array and not
    #  a scalar
    if ndims(new_p) == 0
        new_p = dropdims([new_p], dims=1)
    end

    return Factor(new_dims, new_p)
end
Base.getindex(ϕ::Factor, pair::Pair{NodeName}...) = Base.getindex(ϕ, Assignment(pair))


function Base.setindex!(ϕ::Factor, v, a::Assignment)
    tridx = _translate_index(ϕ, a)
    if Colon() in tridx
        @inbounds return ϕ.potential[tridx...] .= v
    else
        @inbounds return ϕ.potential[tridx...] = v
    end
end

@inline function _translate_index(ϕ::Factor, a::Assignment)
    inds = Array{Any}(undef, ndims(ϕ))
    inds[:] .= Colon()

    for (i, dim) in enumerate(ϕ.dimensions)
        if haskey(a, dim)
            ind = a[dim]

            if isa(ind, Colon)
                continue
            elseif isa(ind, Int)
                if ind < 1 || ind > size(ϕ, dim)
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

# Base.sub2ind(ϕ::Factor, i...) = sub2ind(size(ϕ), i...)
# Base.ind2sub(ϕ::Factor, i) = ind2sub(size(ϕ), i)

