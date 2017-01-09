#
# Test BayesNet generation
#

let
N = 10
mxpa = 4
mxst = 5

bn = rand_discrete_bn(N, mxpa, mxst)

for i = 1:N
    s = Symbol("N", i)
    @test s in names(bn)
end

for nn in names(bn)
    @test ncategories(bn, nn) <= mxst
    @test length(parents(bn, nn)) <= mxpa
end

@test_throws ArgumentError rand_bn_inference(bn, 20, 16)

nq = 3
ne = 5

inf = rand_bn_inference(bn, nq, ne)
@test isempty(intersect(inf.query, keys(inf.evidence)))
end

