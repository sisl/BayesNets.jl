#
# Auxiliary Functions
#
# Not exactly related, but not exactly not

# make sure all dims are valid (in the Factor)
@inline function _check_dims_valid(dims::NodeNames, ϕ::Factor)
    isempty(dims) && return

    dim = first(dims)
    (dim in ϕ) || not_in_factor_error(dim)

    return _check_dims_valid(dims[2:end], ϕ)
end

# dims are unique
_ckeck_dims_unique(dims::NodeNames) = allunique(dims) || non_unique_dims_error()

"""
    duplicate(A, dims)

Repeates an array only through higer dimensions `dims`.

Custom version of repeate, but only outer repetition, and only duplicates
the array for the number of times specified in `dims` for dimensions greater
than `ndims(A)`. If `dims` is empty, returns a copy of `A`.

```jldoctest
julia> duplicate(collect(1:3), 2)
3×2 Array{Int64,2}:
 1  1
 2  2
 3  3

julia> duplicate([1 3; 2 4], 3)
2×2×3 Array{Int64,3}:
[:, :, 1] =
 1  3
 2  4

[:, :, 2] =
 1  3
 2  4

[:, :, 3] =
 1  3
 2  4
```
"""
function duplicate(A::Array, dims::Dims)
    size_in = size(A)

    length_in = length(A)
    size_out = (size_in..., dims...)::Dims

    B = similar(A, size_out)

    # zero in dims means no looping
    @simd for index in 1:prod(dims)
        unsafe_copyto!(B, (index - 1) * length_in + 1, A, 1, length_in)
    end

    return B
end

