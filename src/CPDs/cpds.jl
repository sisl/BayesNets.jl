#=
A CPD is a Conditional Probability Distribution
In general, theyrepresent distribtions of the form P(X|Y)
Each node in a Bayesian Network is associated with a variable,
and contains the CPD relating that var to its parents, P(x | parents(x))
=#

module CPDs

using Reexport
@reexport using Distributions
@reexport using DataFrames

export
    CPD,                         # the abstract CPD type

    Assignment,                  # variable assignment type, complete or partial, for a Bayesian Network
    NodeName,                    # variable name type

    name,                        # obtain the name of the CPD
    distribution,                # returns the CPD's distribution type
    trained,                     # whether the CPD has been trained
    pdf,                         # probability density function or probability distribution function (continuous or discrete)
    learn!                       # train a CPD based on data

import Distributions: ncategories, pdf

typealias NodeName Symbol
typealias Assignment Dict

abstract CPD{D<:Distribution}
#=
Each CPD must implement:
    trained(CPD)
    learn!(CPD, BayesNet, NodeName, DataFrame)
    pdf(CPD, Assignment, parents::AbstractVector{NodeName}) # NOTE: parents must always be in the same topological order

    IF DISCRETE:
        ncategories(CPD)
=#

distribution{D}(cpd::CPD{D}) = D
Base.rand(cpd::CPD, a::Assignment) = rand(pdf(cpd, a))
name(cpd::CPD) = cpd.name # all cpds have names by default

###########################

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

###########################

include("categorical_cpd.jl")
include("linear_gaussian.jl")

end # module CPDs