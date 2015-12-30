let
	bn = BayesNet(
			[BayesNetNode(:A, BINARY_DOMAIN, BernoulliCPD()),
					BayesNetNode(:B, BINARY_DOMAIN, BernoulliCPD([:A],
						                                          Dict(
						                                          	Dict(:A=>true)=>0.1,
						                                          	Dict(:A=>false)=>0.2,
						                                          	   )
						                                         )
					            ),
					],
			[(:A,:B)]
		)

	d = DataFrame(
		A = [true, true, false, false, false],
		B = [true, false, true, false, false],
		)

	#=
	| Row | A     | count |
	|-----|-------|-------|
	| 1   | true  | 2     |
	| 2   | false | 3     |
	=#
	df_count = count(bn, :A, d)
	@test size(df_count) == (2,2)
	@test select(df_count, Dict(:A=>true))[:count] == [2]
	@test select(df_count, Dict(:A=>false))[:count] == [3]

	#=
	| Row | A     | B     | count |
	|-----|-------|-------|-------|
	| 1   | true  | true  | 1     |
	| 2   | true  | false | 1     |
	| 3   | false | true  | 1     |
	| 4   | false | false | 2     |
	=#
	df_count = count(bn, :B, d)
	@test size(df_count) == (4,3)
	@test select(df_count, Dict(:A=>true,  :B=>true))[:count] == [1]
	@test select(df_count, Dict(:A=>true,  :B=>false))[:count] == [1]
	@test select(df_count, Dict(:A=>false, :B=>true))[:count] == [1]
	@test select(df_count, Dict(:A=>false, :B=>false))[:count] == [2]

	# TODO: make this test more rigorous
	@test length(count(bn, d)) == 2

	@test index_data(bn, d) == [2 2 1 1 1;
 							    2 1 2 1 1]

 	#TODO: test this more rigorously
 	@test statistics(bn, d) == Any[
								[3.0 2.0]',
								[2.0 1.0;
								 1.0 1.0]]

	d = rand_table(bn, numSamples = 5)
	@test log_bayes_score(bn, d) < 0.0
end