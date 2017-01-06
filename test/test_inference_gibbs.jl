#
# Test Gibbs sampling and Gibbs inference interface
#

let
bn = rand_discrete_bn()
inf = rand_bn_inference(bn)

ginf = GibbsInferenceState(inf)

@test inf.bn === bn
@test inf.evidence == ginf.evidence
@test inf.query == ginf.query

@test sort(names(ginf.state)) == sort(names(bn))
end

let
bn = rand_discrete_bn()
inf = rand_bn_inference(bn)

ginf = GibbsInferenceState(inf)

# and back again
inf = InferenceState(ginf)

@test inf.bn === bn
@test inf.evidence == ginf.evidence
@test inf.query == ginf.query
end

let
bn = rand_discrete_bn()
inf = GibbsInferenceState(bn, [:N3, :N5, :N2, :N3])

@test inf.query == [:N3, :N5, :N2]
@test sort(names(inf.state)) == sort(names(bn))
end

let
bn = rand_discrete_bn()
inf = GibbsInferenceState(bn, :N6)

@test inf.query == [:N6]
end

let
bn = rand_discrete_bn()

@test_throws ArgumentError GibbsInferenceState(bn, [:N3, :waldo, :N5])
@test_throws ArgumentError GibbsInferenceState(bn, :N1, Assignment(:N1=>2))
end

let
bn = rand_discrete_bn(16, 3, 3);
inf = GibbsInferenceState(bn, [:N3, :N5, :N2, :N3])

@test inf.query == [:N3, :N5, :N2]
end

let
bn = rand_discrete_bn(16, 3, 3);
inf = GibbsInferenceState(bn, :N6)

@test inf.query == [:N6]
end

