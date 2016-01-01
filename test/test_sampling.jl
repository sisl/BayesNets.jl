let
	# A → C ← B
	bn = BayesNet(
			[BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli(1.0)),
					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli(0.0)),
					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
						                                          Dict(
						                                          	Dict(:A=>true,  :B=>true)=>0.1,
						                                          	Dict(:A=>false, :B=>true)=>0.2,
						                                          	Dict(:A=>true,  :B=>false)=>1.0,
						                                          	Dict(:A=>false, :B=>false)=>0.4,
						                                          	   )
						                                         )
					            ),
					],
			[(:A,:C), (:B,:C)]
		)

	@test rand(bn) == Dict(:A=>true, :B=>false, :C=>true)

	t1 = rand_table(bn, numSamples=5)
	@test size(t1) == (5,3)
	@test reduce(&, t1[:A])
	@test !reduce(|, t1[:B])
	@test reduce(&, t1[:C])

	# TODO: actually check whether weighting is being properly done
	t2 = rand_table_weighted(bn, numSamples=5)
	@test size(t1) == (5,3)
	@test reduce(&, t1[:A])
	@test !reduce(|, t1[:B])
	@test reduce(&, t1[:C])

	d1 = rand_bernoulli_dict(2)
	@test length(d1) == 4
	@test haskey(d1, [0,0])
	@test haskey(d1, [0,1])
	@test haskey(d1, [1,0])
	@test haskey(d1, [1,1])

	parentDomains = Array(Vector{Int}, 1)
	parentDomains[1] = [1,2,3]
	d2 = rand_discrete_dict(parentDomains, 2)
	@test length(d2) == 3
	@test isapprox(sum(d2[[1]]), 1.0)
	@test isapprox(sum(d2[[2]]), 1.0)
	@test isapprox(sum(d2[[3]]), 1.0)
end