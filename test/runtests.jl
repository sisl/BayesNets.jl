using BayesNets
using Base.Test

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

@test ne(b.dag) == 3

removeEdges!(b, [(:D, :E), (:C, :D)])

@test ne(b.dag) == 1

