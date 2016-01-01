let
	bn = BayesNet()
	@test isempty(bn.nodes)
	@test isempty(bn.name_to_index)
	@test LightGraphs.ne(bn.dag) == 0
	@test LightGraphs.nv(bn.dag) == 0

	add_node!(bn, BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()))
	@test length(bn.nodes) == 1
	@test bn.name_to_index[:A] == 1

	add_node!(bn, BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()))
	@test length(bn.nodes) == 2
	@test bn.name_to_index[:A] == 1
	@test bn.name_to_index[:B] == 2

	add_edge!(bn, :A, :B)
	@test has_edge(bn, :A, :B)
	@test !has_edge(bn, :B, :A)

	@test node(bn, :A).name == :A
	@test node(bn, :B).name == :B
	@test domain(bn, :B) == BINARY_DOMAIN
	@test sort!(names(bn)) == [:A, :B]
	@test isvalid(bn)
	@test isempty(parents(bn, :A)::Vector{NodeName})
	@test parents(bn, :B) == [:A]

	set_CPD!(bn, :B, CPDs.Bernoulli(m->(m[:A] ? 0.5 : 0.45)))
	my_cpd = cpd(bn, :B)
	@test probvec(my_cpd, Dict(:A=>true)) == [0.5,0.5]
	@test probvec(my_cpd, Dict(:A=>false)) == [0.45,0.55]

	@test isapprox(prob(bn, Dict(:A=>true,  :B=>true)),  0.5*0.5)
	@test isapprox(prob(bn, Dict(:A=>true,  :B=>false)), 0.5*0.5)
	@test isapprox(prob(bn, Dict(:A=>false, :B=>true)),  0.5*0.45)
	@test isapprox(prob(bn, Dict(:A=>false, :B=>false)), 0.5*0.55)

	#=
	Should be:
		A    p
	  true   0.5
	  false  0.5
	=#
	fA = table(bn, :A)
	@test size(fA) == (2,2)
	@test elementwise_isapprox(select(fA, Dict(:A=>true))[:p], [0.5])
	@test elementwise_isapprox(select(fA, Dict(:A=>false))[:p], [0.5])

	#=
	Should be:
	     A      B    p
	  true    true  0.5
	  true   false  0.5
	  false   true  0.45
	  false  false  0.55
	=#
	fB = table(bn, :B)
	@test size(fB) == (4,3)
	@test elementwise_isapprox(select(fB, Dict(:A=>true,  :B=>true))[:p],  [0.5])
	@test elementwise_isapprox(select(fB, Dict(:A=>true,  :B=>false))[:p], [0.5])
	@test elementwise_isapprox(select(fB, Dict(:A=>false, :B=>true))[:p],  [0.45])
	@test elementwise_isapprox(select(fB, Dict(:A=>false, :B=>false))[:p], [0.55])

	#=
	Should be:
	     A      B    p
	  true    true  0.5
	  true   false  0.5
	=#
	fB2 = table(bn, :B, Dict(:A=>true))
	@test size(fB2) == (2,3)
	@test elementwise_isapprox(select(fB2, Dict(:B=>true))[:p],  [0.5])
	@test elementwise_isapprox(select(fB2, Dict(:B=>false))[:p], [0.5])

	#=
	Should be:
	     A      B    p
	  false   true  0.45
	  false  false  0.55
	=#
	fB3 = table(bn, :B, Dict(:A=>false))
	@test size(fB3) == (2,3)
	@test elementwise_isapprox(select(fB3, Dict(:B=>true))[:p],  [0.45])
	@test elementwise_isapprox(select(fB3, Dict(:B=>false))[:p], [0.55])
end

let
	#=
	A → C ← B
	=#

	bn = BayesNet()
	add_nodes!(bn, [BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
						                                          Dict(
						                                          	Dict(:A=>true,  :B=>true)=>0.1,
						                                          	Dict(:A=>false, :B=>true)=>0.2,
						                                          	Dict(:A=>true,  :B=>false)=>0.3,
						                                          	Dict(:A=>false, :B=>false)=>0.4,
						                                          	   )
						                                         )
					            ),
					])
	add_edges!(bn, [(:A,:C), (:B,:C)])

	@test sort!(names(bn)) == [:A, :B, :C]
	@test isempty(parents(bn, :A))
	@test isempty(parents(bn, :B))
	@test sort!(parents(bn, :C)) == [:A, :B]
	@test isvalid(bn)

	@test size(table(bn, :C)) == (8,4)

	remove_edges!(bn, [(:A,:C), (:B,:C)])
	@test isempty(parents(bn, :A))
	@test isempty(parents(bn, :B))
	@test isempty(parents(bn, :C))
	@test isvalid(bn)

	set_domain!(bn, :A, ContinuousDomain(0.0, 1.0))
	@test isa(domain(bn, :A), ContinuousDomain)
end

let
	bn = BayesNet([:A, :B, :C])
	add_edges!(bn, [(:A,:C), (:B,:C)])

	@test sort!(names(bn)) == [:A, :B, :C]
	@test isempty(parents(bn, :A))
	@test isempty(parents(bn, :B))
	@test sort!(parents(bn, :C)) == [:A, :B]
	@test isvalid(bn)
end

let
	bn = BayesNet(
			DAG(3), [BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
						                                          Dict(
						                                          	Dict(:A=>true,  :B=>true)=>0.1,
						                                          	Dict(:A=>false, :B=>true)=>0.2,
						                                          	Dict(:A=>true,  :B=>false)=>0.3,
						                                          	Dict(:A=>false, :B=>false)=>0.4,
						                                          	   )
						                                         )
					            ),
					]
		)
	add_edges!(bn, [(:A,:C), (:B,:C)])

	@test sort!(names(bn)) == [:A, :B, :C]
	@test isempty(parents(bn, :A))
	@test isempty(parents(bn, :B))
	@test sort!(parents(bn, :C)) == [:A, :B]
	@test isvalid(bn)
end

let
	bn = BayesNet(
			[BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
						                                          Dict(
						                                          	Dict(:A=>true,  :B=>true)=>0.1,
						                                          	Dict(:A=>false, :B=>true)=>0.2,
						                                          	Dict(:A=>true,  :B=>false)=>0.3,
						                                          	Dict(:A=>false, :B=>false)=>0.4,
						                                          	   )
						                                         )
					            ),
					],
			[(:A,:C), (:B,:C)]
		)

	ordering = topological_sort_by_dfs(bn.dag)
	@test ordering == [1,2,3] || ordering == [2,1,3]
	@test sort!(names(bn)) == [:A, :B, :C]
	@test isempty(parents(bn, :A))
	@test isempty(parents(bn, :B))
	@test sort!(parents(bn, :C)) == [:A, :B]
	@test isvalid(bn)
end

let
	bn = BayesNet([:B, :S, :E, :D, :C]) # construct an edgeless network with five variables
                                    # note that this defaults to binary variables with 50/50 Bernoulli CPDs
	add_edges!(bn, [(:B, :E), (:S, :E), (:E, :D), (:E, :C)]) # add edges, does not change CPDs

	set_CPD!(bn, :B, CPDs.Bernoulli(0.1))
	set_CPD!(bn, :S, CPDs.Bernoulli(0.5))
	set_CPD!(bn, :E, CPDs.Bernoulli([:B, :S], rand_bernoulli_dict(2)))
	set_CPD!(bn, :D, CPDs.Bernoulli([:E], rand_bernoulli_dict(1)))
	set_CPD!(bn, :C, CPDs.Bernoulli([:E], rand_bernoulli_dict(1)))

	table(bn, :D) # ensure that this doens't fail
end