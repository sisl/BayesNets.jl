#
# Test for integration between Factors.jl and BayesNets
#

let
bn = rand_discrete_bn(10, 4)
name = :N5

ft = Factor(bn, name)
df = join(df = DataFrame(ft), table(bn, name), on=names(ft))
diff = abs(df[:p] - df[:v])

@test any(diff .> 1E-10)
end

