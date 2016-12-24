#
# Auxiliary Functions
#
# Not exactly related, but not exactly not

"""
    duplicate(A, dims)

Repeates an array only through higer dimensions `dims`.

Custom version of repeate, but only outer repetition, and only duplicates
the array for the number of times specified in `dims` for dimensions greater
than `ndims(A)`

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
function duplicate(A::Array, dims)
    size_in = size(A)

    length_in = length(A)
    size_out = (size_in..., dims...)::Dims

    B = similar(A, size_out)

    # zero in dims means no looping
    @simd for index in 1:prod(dims)
        unsafe_copy!(B, (index - 1) * length_in + 1, A, 1, length_in)
    end

    return B
end

