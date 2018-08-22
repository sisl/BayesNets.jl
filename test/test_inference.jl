#
# Test inference setup/interface
#

let
    bn = rand_discrete_bn()

    @test_throws ArgumentError InferenceState(bn, [:waldo, :N3])
    @test_throws ArgumentError InferenceState(bn, [:N2, :N3], Assignment(:N1 => 1, :N3 => 3, :N7 => 2016))
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


let
    # A → C ← B
    bn = DiscreteBayesNet()
    push!(bn, DiscreteCPD(:a, [1.0,0.0]))
    push!(bn, DiscreteCPD(:b, [0.0,1.0]))
    push!(bn, DiscreteCPD(:c, [:a, :b], [2,2], [Categorical([0.1,0.9]),
                                                Categorical([0.2,0.8]),
                                                Categorical([1.0,0.0]),
                                                Categorical([0.4,0.6]),
                                                ]))

    for im in [ExactInference(), LikelihoodWeightingInference(), LoopyBelief(), GibbsSamplingNodewise(), GibbsSamplingFull()]
        Random.seed!(0)
        ϕ = infer(im, bn, :a)::Factor
        @test length(ϕ) == 2
        f = ϕ[:a=>1]::Factor
        @test length(f) == 1
        @test isapprox(f.potential[1], 1.0, atol=0.02)

        f = ϕ[:a=>2]::Factor
        @test length(f) == 1
        @test isapprox(f.potential[1], 0.0, atol=0.02)
    end

    for im in [ExactInference(), LikelihoodWeightingInference(), LoopyBelief(), GibbsSamplingNodewise(), GibbsSamplingFull()]
        Random.seed!(0)
        ϕ = infer(im, bn, :c)::Factor
        @test length(ϕ) == 2
        @test isapprox(ϕ[:c=>1].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:c=>2].potential[1], 0.0, atol=0.02)
    end

    for im in [ExactInference(), LikelihoodWeightingInference(), GibbsSamplingNodewise(), GibbsSamplingFull()] # LoopyBelief(),
        Random.seed!(0)
        ϕ = infer(im, bn, [:b, :c])
        @test size(ϕ) == (2,2)
        @test isapprox(ϕ[:b=>1, :c=>1].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>2, :c=>1].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:b=>1, :c=>2].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>2, :c=>2].potential[1], 0.0, atol=0.02)
    end

    # Student Example (PGM page 53)
    bn = DiscreteBayesNet()
    push!(bn, DiscreteCPD(:D, [0.6,0.4])) # difficulty
    push!(bn, DiscreteCPD(:I, [0.7,0.3])) # intelligence
    push!(bn, DiscreteCPD(:G, [:D, :I], [2,2], [Categorical([0.3,0.4,0.3]), # grade
                                                Categorical([0.9,0.08,0.02]),
                                                Categorical([0.05,0.25,0.7]),
                                                Categorical([0.5,0.3,0.2]),
                                                ]))
    push!(bn, DiscreteCPD(:L, [:G], [3], [Categorical([0.1,0.9]), # letter
                                          Categorical([0.4,0.6]),
                                          Categorical([0.99,0.01]),
                                         ]))
    push!(bn, DiscreteCPD(:S, [:I], [2], [Categorical([0.95,0.05]), # SAT
                                          Categorical([0.2,0.8]),
                                         ]))

    # P(D) = [0.6, 0.4]
    for im in [ExactInference(), LikelihoodWeightingInference(), LoopyBelief(), GibbsSamplingNodewise(), GibbsSamplingFull()]
        Random.seed!(0)
        ϕ = infer(im, bn, :D)
        @test size(ϕ) == (2,)
        @test isapprox(ϕ[:D=>1].potential[1], 0.6, atol=0.05)
        @test isapprox(ϕ[:D=>2].potential[1], 0.4, atol=0.05)
    end

    # P(G|d₁, i₁) = [0.3, 0.4, 0.3]
    for im in [ExactInference(), LikelihoodWeightingInference(), LoopyBelief(), GibbsSamplingNodewise(), GibbsSamplingFull()]
        Random.seed!(0)
        ϕ = infer(im, bn, :G, evidence=Assignment(:D=>1, :I=>1))
        @test size(ϕ) == (3,)
        @test isapprox(ϕ.potential[1], 0.3, atol=0.05)
        @test isapprox(ϕ.potential[2], 0.4, atol=0.05)
        @test isapprox(ϕ.potential[1], 0.3, atol=0.05)
    end
end
