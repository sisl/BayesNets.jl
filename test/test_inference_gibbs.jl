#
# Test Gibbs sampling and Gibbs inference interface
#

let
bn = rand_discrete_bn(16, 3, 3);
(qu, ev) = bn_inference_init(bn, 2, 3)

inf = InferenceState(bn, qu, ev)
ginf = GibbsInferenceState(bn, qu, ev)

@test evidence(ginf) == ev
@test query(ginf) == qu

@test inf.bn === ginf.bn
@test inf.factor === ginf.factor

# state has state for all things to have state for
@test sort(names(ginf.state)) == sort(names(ginf))
@test sort(names(ginf.state)) == sort(names(inf))

# correctly converting between the two
@test evidence(inf) == evidence(convert(InferenceState, ginf))
@test query(inf) == query(convert(InferenceState, ginf))
@test inf.bn === convert(InferenceState, ginf).bn
@test inf.factor === convert(InferenceState, ginf).factor

@test evidence(ginf) == evidence(convert(GibbsInferenceState, inf))
@test query(ginf) == query(convert(GibbsInferenceState, inf))
@test ginf.bn === convert(GibbsInferenceState, inf).bn
@test ginf.factor === convert(GibbsInferenceState, inf).factor
end

