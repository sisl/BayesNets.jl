#
# Test inference setup/interface
#

let
bn = rand_discrete_bn()

@test_throws ArgumentError InferenceState(bn, [:waldo, :N3])
@test_throws ArgumentError InferenceState(bn, [:N2, :N3],
        Assignment(:N1 => 1, :N3 => 3, :N7 => 2016))
end

let
bn = rand_discrete_bn()
inf = InferenceState(bn, [:N3, :N5, :N2, :N3])

@test inf.query == [:N3, :N5, :N2]
end

let
bn = rand_discrete_bn()
inf = InferenceState(bn, :N6)

@test inf.query == [:N6]
end

let
bn = rand_discrete_bn()

@test_throws ArgumentError InferenceState(bn, [:N3, :waldo, :N5])
@test_throws ArgumentError InferenceState(bn, :N1, Assignment(:N1=>2))
end

