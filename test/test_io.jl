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

	# TODO: make this test more rigorous
	assignments = BayesNets.assignment_dicts(bn, [:A, :B])
	@test assignments ==
			[Dict{Symbol,Any}(:B=>false,:A=>false),
			 Dict{Symbol,Any}(:B=>false,:A=>true),
			 Dict{Symbol,Any}(:B=>true,:A=>false),
			 Dict{Symbol,Any}(:B=>true,:A=>true)]

	# TODO: make this test more rigorous
	@test BayesNets.discrete_parameter_dict(
				BayesNets.assignment_dicts(bn, [:A]),
				[0.2, 0.8, 0.4, 0.6], 2
				) ==
		Dict(Dict{Symbol,Any}(:A=>false)=>[0.2,0.8],
			 Dict{Symbol,Any}(:A=>true)=>[0.4,0.6])

	# TODO: properly test readxdsl
end