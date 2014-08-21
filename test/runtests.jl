using BayesNets
using Base.Test

b = BayesNet([:A, :B, :C])
addEdge!(b, :A, :B)
setCPD!(b, :A, CPDs.Bernoulli(0.5))
setCPD!(b, :B, CPDs.Bernoulli(m->(m[:A] ? 0.5 : 0.45)))
setCPD!(b, :C, CPDs.Bernoulli(0.5))

@test length(b.names) == 3

d = randTable(b, numSamples = 5)
@test size(d, 1) == 5

