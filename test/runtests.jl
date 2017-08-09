using BayesNets
using DataFrames
using Base.Test
using LightGraphs

srand(0)

"""
A simple variant of isapprox that is true if the isapprox comparison works
elementwise in the vector
"""
function elementwise_isapprox{F<:AbstractFloat}(x::AbstractArray{F},
        y::AbstractArray{F},
        rtol::F=sqrt(eps(F)),
        atol::F=zero(F))

    if length(x) != length(y)
        return false
    end

    for (a,b) in zip(x,y)
        if !isapprox(a,b,rtol=rtol, atol=atol)
            return false
        end
    end

    true
end

testdir = joinpath(dirname(@__DIR__), "test")
include(joinpath(testdir, "test_utils.jl"))
include(joinpath(testdir, "test_cpds.jl"))
include(joinpath(testdir, "test_tables.jl"))
include(joinpath(testdir, "test_factors.jl"))
include(joinpath(testdir, "test_bayesnets.jl"))
include(joinpath(testdir, "test_gibbs.jl"))
include(joinpath(testdir, "test_sampling.jl"))
include(joinpath(testdir, "test_inference.jl"))
include(joinpath(testdir, "test_learning.jl"))
include(joinpath(testdir, "test_io.jl"))
include(joinpath(testdir, "test_ndgrid.jl"))

include(joinpath(testdir, "test_discrete_bayes_nets.jl"))
include(joinpath(testdir, "test_genbn.jl"))

include(joinpath(testdir, "test_docs.jl"))
