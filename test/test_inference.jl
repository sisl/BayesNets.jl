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

    for im in [ExactInference(), ] # LikelihoodWeightingInference(), LoopyBelief()
        srand(0)
        ϕ = infer(im, bn, :a)
        @test size(ϕ) == (2,2)
        @test isapprox(select(ϕ, Assignment(:a=>1))[:p][1], 1.0, atol=0.02)
        @test isapprox(select(ϕ, Assignment(:a=>2))[:p][1], 0.0, atol=0.02)
    end

    for im in [ExactInference(), ] # LikelihoodWeightingInference(), LoopyBelief()
        srand(0)
        ϕ = infer(im, bn, :c)
        @test size(ϕ) == (2,2)
        @test isapprox(select(ϕ, Assignment(:c=>1))[:p][1], 1.0, atol=0.02)
        @test isapprox(select(ϕ, Assignment(:c=>2))[:p][1], 0.0, atol=0.02)
    end

    for im in [ExactInference(), ] # LikelihoodWeightingInference(), LoopyBelief(),
        srand(0)
        ϕ = infer(im, bn, [:b, :c])
        @test size(ϕ) == (4,3)
        @test isapprox(select(ϕ, Assignment(:b=>1, :c=>1))[:p][1], 0.0, atol=0.02)
        @test isapprox(select(ϕ, Assignment(:b=>2, :c=>1))[:p][1], 1.0, atol=0.02)
        @test isapprox(select(ϕ, Assignment(:b=>1, :c=>2))[:p][1], 0.0, atol=0.02)
        @test isapprox(select(ϕ, Assignment(:b=>2, :c=>2))[:p][1], 0.0, atol=0.02)
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
    for im in [ExactInference(), LikelihoodWeightingInference(), ] # LoopyBelief()
        srand(0)
        ϕ = infer(im, bn, :D)
        @test size(ϕ) == (2,2)
        @test isapprox(select(ϕ, Assignment(:D=>1))[:p][1], 0.6, atol=0.05)
        @test isapprox(select(ϕ, Assignment(:D=>2))[:p][1], 0.4, atol=0.05)
    end

    # P(G|d₁, i₁) = [0.3, 0.4, 0.3]
    for im in [ExactInference(), LikelihoodWeightingInference(), ] # LoopyBelief()
        srand(0)
        ϕ = infer(im, bn, :G, evidence=Assignment(:D=>1, :I=>1))
        @test size(ϕ) == (3,2)
        @test isapprox(select(ϕ, Assignment(:G=>1))[:p][1], 0.3, atol=0.05)
        @test isapprox(select(ϕ, Assignment(:G=>2))[:p][1], 0.4, atol=0.05)
        @test isapprox(select(ϕ, Assignment(:G=>3))[:p][1], 0.3, atol=0.05)
    end

end