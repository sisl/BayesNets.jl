type FakeScoringFunction <: ScoringFunction end
type FakeGraphSearchStrategy <: GraphSearchStrategy end

let
	function test_disc_bn(bn::DiscreteBayesNet)
		#=
		│ Row │ A │ count │
		├─────┼───┼───────┤
		│ 1   │ 1 │ 4     │
		│ 2   │ 2 │ 4     │
		│ 3   │ 3 │ 4     │
		=#
		df_count = count(bn, :A, data)
		@test size(df_count) == (3,2)
		@test select(df_count, Assignment(:A=>1))[:count] == [4]
		@test select(df_count, Assignment(:A=>2))[:count] == [4]

		#=
		│ Row │ A │ B │ C │ count │
		├─────┼───┼───┼───┼───────┤
		│ 1   │ 1 │ 1 │ 1 │ 3     │
		│ 2   │ 1 │ 2 │ 2 │ 1     │
		│ 3   │ 2 │ 2 │ 1 │ 2     │
		│ 4   │ 2 │ 2 │ 2 │ 1     │
		│ 5   │ 2 │ 1 │ 1 │ 1     │
		│ 6   │ 3 │ 1 │ 1 │ 3     │
		│ 7   │ 3 │ 2 │ 2 │ 1     │
		=#
		df_count = count(bn, :C, data)
		@test size(df_count) == (7,4)
		@test select(df_count, Assignment(:A=>1, :B=>1))[:count] == [3]
		@test select(df_count, Assignment(:A=>1, :B=>2))[:count] == [1]
		@test select(df_count, Assignment(:A=>2, :B=>1))[:count] == [1]
		@test select(df_count, Assignment(:A=>2, :B=>2))[:count] == [2,1]
	end

	data = DataFrame(A=[1,1,1,1,2,2,2,2,3,3,3,3],
                     B=[1,1,1,2,2,2,2,1,1,2,1,1],
                     C=[1,1,1,2,1,1,2,1,1,2,1,1])

	dag = DAG(3)
	add_edge!(dag, 1, 2)
	add_edge!(dag, 1, 3)
	add_edge!(dag, 2, 3)

	bn = fit(BayesNet, data, dag, [DiscreteCPD, DiscreteCPD, DiscreteCPD])
	bn = fit(BayesNet, data, (:A=>:B, :A=>:C, :B=>:C), [DiscreteCPD, DiscreteCPD, DiscreteCPD])
	bn = fit(DiscreteBayesNet, data, :A=>:B)
	bn = fit(BayesNet, data, :A=>:B, [DiscreteCPD, DiscreteCPD, DiscreteCPD])
	bn = fit(BayesNet, data, :A=>:B, DiscreteCPD)

	bn = fit(DiscreteBayesNet, data, (:A=>:B, :A=>:C, :B=>:C))
	test_disc_bn(bn)
	bn = fit(DiscreteBayesNet, data, dag)
	test_disc_bn(bn)

	@test_throws ErrorException score_component(FakeScoringFunction(), StaticCPD(:a, Bernoulli(0.5)), data)

	cache = ScoreComponentCache(data)
	score = score_component(NegativeBayesianInformationCriterion(), StaticCPD(:A, Categorical(3)), data)
	@test isapprox(score, -33.82141487739863)
	score = score_component(NegativeBayesianInformationCriterion(), StaticCPD(:A, Categorical(3)), data, cache)
	@test isapprox(score, -33.82141487739863)

	@test_throws ErrorException fit(DiscreteBayesNet, data, FakeGraphSearchStrategy())

	K2 = K2GraphSearch([:A,:B,:C], [DiscreteCPD, DiscreteCPD, DiscreteCPD])
	K22 = K2GraphSearch([:A,:B,:C], [DiscreteCPD])

	bn3 = fit(DiscreteBayesNet, data, K2)
	bn4 = fit(BayesNet, data, K2)

	# TODO: make this test more rigorous
	@test length(count(bn, data)) == 3

	@test statistics(bn, :A, data) == reshape([4, 4, 4], (3,1))
	@test statistics(bn, :C, data) == [3 1 3 0 2 0;
 							           0 0 0 1 1 1]

 	#TODO: test this more rigorously
 	@test statistics(bn, data) == Matrix{Int}[
 									statistics(bn, :A, data),
 									statistics(bn, :B, data),
 									statistics(bn, :C, data),
								]

	data = rand(bn, 5)
	@test bayesian_score(bn, data) < 0.0
end