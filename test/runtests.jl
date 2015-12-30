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

include("test_assignments.jl")
include("test_factors.jl")
include("test_cpds.jl")

# TODO(tim): move the stuff below

b = BayesNet([:A, :B, :C, :D, :E])
addEdge!(b, :A, :B)
setCPD!(b, :A, BernoulliCPD(0.5))
setCPD!(b, :B, BernoulliCPD(m->(m[:A] ? 0.5 : 0.45)))
setCPD!(b, :C, BernoulliCPD(0.5))

@test length(b.names) == 5

addEdges!(b, [(:A, :C), (:D, :E), (:C, :D)])

d = randTable(b, numSamples = 5)
@test size(d, 1) == 5

removeEdge!(b, :A, :C)

@test LightGraphs.ne(b.dag) == 3

removeEdges!(b, [(:D, :E), (:C, :D)])

@test LightGraphs.ne(b.dag) == 1

# Code from documentation

b = BayesNet([:B, :S, :E, :D, :C])
addEdges!(b, [(:B, :E), (:S, :E), (:E, :D), (:E, :C)])

setCPD!(b, :B, BernoulliCPD(0.1))
setCPD!(b, :S, BernoulliCPD(0.5))
setCPD!(b, :E, BernoulliCPD([:B, :S], randBernoulliDict(2)))
setCPD!(b, :D, BernoulliCPD([:E], randBernoulliDict(1)))
setCPD!(b, :C, BernoulliCPD([:E], randBernoulliDict(1)));

@test isequal(parents(b, :E), [:B, :S])

@test domain(b, :C).elements[1] == false
@test domain(b, :C).elements[2] == true

@test size(table(b, :D)) == (4,3)
@test size(table(b, :B)) == (2,2)

bt = table(b, :B)
st = table(b, :S)
et = table(b, :E)
tt = bt * et * st

@test size(tt) == (8,4)
tt = sumout(tt, [:B, :S])
@test size(tt) == (2,2)
a = Dict(
    :B=>0,
    :S=>1,
    :E=>0,
    :D=>1,
    :C=>1
)
@test prob(b, a) <= 1.0
@test length(rand(b)) == 5
@test size(randTable(b, numSamples=5)) == (5,5)
t = randTable(b, numSamples=10, consistentWith=Dict(:B=>true, :C=>false))
@test size(t,1) <= 10
@test size(t,2) == 5
t = randTable(b, numSamples=100, consistentWith=Dict(:B=>true, :C=>false))
@test size(estimate(t),2) == 6

b = BayesNet([:A, :B, :C])
addEdge!(b, :A, :B)
setCPD!(b, :A, BernoulliCPD(0.5))
setCPD!(b, :B, BernoulliCPD(m->(m[:A] ? 0.5 : 0.45)))
setCPD!(b, :C, BernoulliCPD(0.5))

d = randTable(b, numSamples = 5)
@test size(d) == (5,3)
count(b, d)
@test logBayesScore(b, d) < 0.0
