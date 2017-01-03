#
# Test inference setup/interface
#

let
bn = rand_discrete_bn(16, 3, 3);
(qu, ev) = rand_bn_inference(bn, 2, 3)

inf = InferenceState(bn, qu, ev)
ginf = GibbsInferenceState(bn, qu, ev)

@test inf.bn === bn
@test evidence(inf) == ev
@test query(inf) == qu
end

