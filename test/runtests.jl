using BayesNets
using DataFrames
using Base.Test


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

include("test_utils.jl")




b = BayesNet([:A, :B, :C, :D, :E])
addEdge!(b, :A, :B)
setCPD!(b, :A, CPDs.Bernoulli(0.5))
setCPD!(b, :B, CPDs.Bernoulli(m->(m[:A] ? 0.5 : 0.45)))
setCPD!(b, :C, CPDs.Bernoulli(0.5))

@test length(b.names) == 5

addEdges!(b, [(:A, :C), (:D, :E), (:C, :D)])

d = randTable(b, numSamples = 5)
@test size(d, 1) == 5

removeEdge!(b, :A, :C)

@test LightGraphs.ne(b.dag) == 3

removeEdges!(b, [(:D, :E), (:C, :D)])

@test LightGraphs.ne(b.dag) == 1

