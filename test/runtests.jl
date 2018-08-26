using BayesNets
using DataFrames
using Test
using LightGraphs
using Statistics
using LinearAlgebra
using Random


Random.seed!(0)

"""
A simple variant of isapprox that is true if the isapprox comparison works
elementwise in the vector
"""
function elementwise_isapprox(x::AbstractArray{F},
                              y::AbstractArray{F},
                              rtol::F=sqrt(eps(F)),
                              atol::F=zero(F)) where {F<:AbstractFloat}

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

@testset "bn" begin
    testdir = joinpath(dirname(@__DIR__), "test")
    @testset "utils" begin
        include(joinpath(testdir, "test_utils.jl"))
    end
    @testset "cpds" begin
        include(joinpath(testdir, "test_cpds.jl"))
    end
    @testset "tables" begin
        include(joinpath(testdir, "test_tables.jl"))
    end
    @testset "factors" begin
        include(joinpath(testdir, "test_factors.jl"))
    end
    @testset "bayesnets" begin
        include(joinpath(testdir, "test_bayesnets.jl"))
    end
    @testset "gibbs" begin
        include(joinpath(testdir, "test_gibbs.jl"))
    end
    @testset "sampling" begin
        include(joinpath(testdir, "test_sampling.jl"))
    end
    @testset "inference" begin
        include(joinpath(testdir, "test_inference.jl"))
    end
    @testset "learning" begin
        include(joinpath(testdir, "test_learning.jl"))
    end
    @testset "io" begin
        include(joinpath(testdir, "test_io.jl"))
    end
    @testset "ndgrid" begin
        include(joinpath(testdir, "test_ndgrid.jl"))
    end
    @testset "discrete bayes" begin
        include(joinpath(testdir, "test_discrete_bayes_nets.jl"))
    end
    @testset "gen bn" begin
        include(joinpath(testdir, "test_genbn.jl"))
    end
    @testset "docs" begin
        include(joinpath(testdir, "test_docs.jl"))
    end
end
