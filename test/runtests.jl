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

include("test_cpds.jl")
# include("test_factors.jl")
# include("test_bayesnets.jl")
# include("test_sampling.jl")
# include("test_learning.jl")
# include("test_io.jl")

include("test_discrete_bayes_nets.jl")