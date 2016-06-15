"""
The ordering of the parental instantiations in discrete networks follows the convention
defined in Decision Making Under Uncertainty.

Suppose a variable has three discrete parents. The first parental instantiation
assigns all parents to their first bin. The second will assign the first
parent (as defined in `parents`) to its second bin and the other parents
to their first bin. The sequence continues until all parents are instantiated
to their last bins.

This is a directly copy from Base.sub2ind but allows for passing a vector instead of separate items

Note that this does NOT check bounds
"""
function sub2ind_vec{T<:Integer}(dims::Tuple{Vararg{Integer}}, I::AbstractVector{T})
    N = length(dims)
    @assert(N == length(I))

    ex = I[N] - 1
    for i in N-1:-1:1
        if i > N
            ex = (I[i] - 1 + ex)
        else
            ex = (I[i] - 1 + dims[i]*ex)
        end
    end

    ex + 1
end

"""
Infer the number of instantiations, N, for a data type, assuming
that it takes on the values 1:N
"""
function infer_number_of_instantiations{I<:Int}(arr::AbstractVector{I})
    lo, hi = extrema(arr)
    lo ≥ 1 || error("infer_number_of_instantiations assumes values in 1:N, value $lo found!")
    lo == 1 || warn("infer_number_of_instantiations assumes values in 1:N, lowest value is $(lo)!")
    hi
end