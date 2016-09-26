using BayesNets
using DataFrames
using Base.Test
using LightGraphs

"""
A simple variant of isapprox that is true if the isapprox comparison works
elementwise in the vector
"""
function elementwise_isapprox{F<:AbstractFloat}(x::AbstractArray{F}, y::AbstractArray{F},
	rtol::F=sqrt(eps(F)),
	atol::F=zero(F),
	)

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

include(Pkg.dir("BayesNets", "test", "test_utils.jl"))
include(Pkg.dir("BayesNets", "test", "test_cpds.jl"))
include(Pkg.dir("BayesNets", "test", "test_factors.jl"))
include(Pkg.dir("BayesNets", "test", "test_bayesnets.jl"))
include(Pkg.dir("BayesNets", "test", "test_sampling.jl"))
include(Pkg.dir("BayesNets", "test", "test_learning.jl"))
include(Pkg.dir("BayesNets", "test", "test_io.jl"))
include(Pkg.dir("BayesNets", "test", "test_ndgrid.jl"))

include(Pkg.dir("BayesNets", "test", "test_discrete_bayes_nets.jl"))

include(Pkg.dir("BayesNets", "test", "test_docs.jl"))