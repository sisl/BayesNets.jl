#
# Test for integration between Factors.jl and BayesNets
#

let
bn = rand_discrete_bn(6, 2)
name = :N5

ft = Factor(bn, name)

# table puts the actual dimension last, parents first
nd = ndims(ft)
permutedims!(ft, vcat(nd, 1:(nd-1)))
df = DataFrame(ft)
rename!(df, :v, :p)

@test df == table(bn, name)
end

