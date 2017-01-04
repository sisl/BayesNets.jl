#
# Test inference setup/interface
#

let
bn = rand_discrete_bn(16, 3, 3);
inf = rand_bn_inference(bn, 2, 3)

ginf = GibbsInferenceState(inf)

@test inf.bn === bn
@test inf.evidence == ginf.evidence
@test inf.query == ginf.query
end

