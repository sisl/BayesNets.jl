let
	bn = BayesNet()
	@test length(bn) == 0

	push!(bn, StaticCPD(CPDCore(:a, NodeName[], Normal(0.0,1.0))))

	@test length(bn) == 1
	@test name(get(bn, :a)) == :a
	@test name(get(bn, 1)) == :a
	@test names(bn) == [:a]
	@test parents(bn, :a) == NodeName[]
	@test children(bn, :a) == NodeName[]
	@test pdf!(bn, Dict(:a=>0.0)) == pdf(Normal(0.0,1.0), 0.0)
	@test logpdf!(bn, Dict(:a=>0.0)) == logpdf(Normal(0.0,1.0), 0.0)
	@test isapprox(pdf!(bn, DataFrame(a=[0.0, 1.0])), pdf(Normal(0.0,1.0), 0.0) * pdf(Normal(0.0,1.0), 1.0))
	@test isapprox(logpdf!(bn, DataFrame(a=[0.0, 1.0])), logpdf(Normal(0.0,1.0), 0.0) + logpdf(Normal(0.0,1.0), 1.0))

	# b = 2a + 1
	push!(bn, LinearGaussianCPD(CPDCore(:b, [:a], Normal(0.0,1.0)), [2.0], 1.0))
	@test length(bn) == 2
	@test name(get(bn, :b)) == :b
	@test name(get(bn, 2)) == :b
	@test has_edge(bn, :a, :b)
	@test names(bn) == [:a, :b]
	@test parents(bn, :a) == NodeName[]
	@test children(bn, :a) == [:b]
	@test parents(bn, :b) == [:a]
	@test children(bn, :b) == NodeName[]
	@test pdf!(bn, Dict(:a=>1.0, :b=>2.0)) == pdf(Normal(0.0,1.0), 1.0) * pdf(Normal(3.0, 1.0), 2.0)
	@test logpdf!(bn, Dict(:a=>1.0, :b=>2.0)) == logpdf(Normal(0.0,1.0), 1.0) + logpdf(Normal(3.0, 1.0), 2.0)
end

# 	#=
# 	Should be:
# 		A    p
# 	  true   0.5
# 	  false  0.5
# 	=#
# 	fA = table(bn, :A)
# 	@test size(fA) == (2,2)
# 	@test elementwise_isapprox(select(fA, Dict(:A=>true))[:p], [0.5])
# 	@test elementwise_isapprox(select(fA, Dict(:A=>false))[:p], [0.5])

# 	#=
# 	Should be:
# 	     A      B    p
# 	  true    true  0.5
# 	  true   false  0.5
# 	  false   true  0.45
# 	  false  false  0.55
# 	=#
# 	fB = table(bn, :B)
# 	@test size(fB) == (4,3)
# 	@test elementwise_isapprox(select(fB, Dict(:A=>true,  :B=>true))[:p],  [0.5])
# 	@test elementwise_isapprox(select(fB, Dict(:A=>true,  :B=>false))[:p], [0.5])
# 	@test elementwise_isapprox(select(fB, Dict(:A=>false, :B=>true))[:p],  [0.45])
# 	@test elementwise_isapprox(select(fB, Dict(:A=>false, :B=>false))[:p], [0.55])

# 	#=
# 	Should be:
# 	     A      B    p
# 	  true    true  0.5
# 	  true   false  0.5
# 	=#
# 	fB2 = table(bn, :B, Dict(:A=>true))
# 	@test size(fB2) == (2,3)
# 	@test elementwise_isapprox(select(fB2, Dict(:B=>true))[:p],  [0.5])
# 	@test elementwise_isapprox(select(fB2, Dict(:B=>false))[:p], [0.5])

# 	#=
# 	Should be:
# 	     A      B    p
# 	  false   true  0.45
# 	  false  false  0.55
# 	=#
# 	fB3 = table(bn, :B, Dict(:A=>false))
# 	@test size(fB3) == (2,3)
# 	@test elementwise_isapprox(select(fB3, Dict(:B=>true))[:p],  [0.45])
# 	@test elementwise_isapprox(select(fB3, Dict(:B=>false))[:p], [0.55])
# end

# let
# 	#=
# 	A → C ← B
# 	=#

# 	bn = BayesNet()
# 	add_nodes!(bn, [BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
# 						                                          Dict(
# 						                                          	Dict(:A=>true,  :B=>true)=>0.1,
# 						                                          	Dict(:A=>false, :B=>true)=>0.2,
# 						                                          	Dict(:A=>true,  :B=>false)=>0.3,
# 						                                          	Dict(:A=>false, :B=>false)=>0.4,
# 						                                          	   )
# 						                                         )
# 					            ),
# 					])
# 	add_edges!(bn, [(:A,:C), (:B,:C)])

# 	@test sort!(names(bn)) == [:A, :B, :C]
# 	@test isempty(parents(bn, :A))
# 	@test isempty(parents(bn, :B))
# 	@test sort!(parents(bn, :C)) == [:A, :B]
# 	@test isvalid(bn)

# 	@test size(table(bn, :C)) == (8,4)

# 	remove_edges!(bn, [(:A,:C), (:B,:C)])
# 	@test isempty(parents(bn, :A))
# 	@test isempty(parents(bn, :B))
# 	@test isempty(parents(bn, :C))
# 	@test isvalid(bn)

# 	set_domain!(bn, :A, ContinuousDomain(0.0, 1.0))
# 	@test isa(domain(bn, :A), ContinuousDomain)
# end

# let
# 	bn = BayesNet([:A, :B, :C])
# 	add_edges!(bn, [(:A,:C), (:B,:C)])

# 	@test sort!(names(bn)) == [:A, :B, :C]
# 	@test isempty(parents(bn, :A))
# 	@test isempty(parents(bn, :B))
# 	@test sort!(parents(bn, :C)) == [:A, :B]
# 	@test isvalid(bn)
# end

# let
# 	bn = BayesNet(
# 			DAG(3), [BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
# 						                                          Dict(
# 						                                          	Dict(:A=>true,  :B=>true)=>0.1,
# 						                                          	Dict(:A=>false, :B=>true)=>0.2,
# 						                                          	Dict(:A=>true,  :B=>false)=>0.3,
# 						                                          	Dict(:A=>false, :B=>false)=>0.4,
# 						                                          	   )
# 						                                         )
# 					            ),
# 					]
# 		)
# 	add_edges!(bn, [(:A,:C), (:B,:C)])

# 	@test sort!(names(bn)) == [:A, :B, :C]
# 	@test isempty(parents(bn, :A))
# 	@test isempty(parents(bn, :B))
# 	@test sort!(parents(bn, :C)) == [:A, :B]
# 	@test isvalid(bn)
# end

# let
# 	bn = BayesNet(
# 			[BayesNetNode(:A, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:B, BINARY_DOMAIN, CPDs.Bernoulli()),
# 					BayesNetNode(:C, BINARY_DOMAIN, CPDs.Bernoulli([:A, :B],
# 						                                          Dict(
# 						                                          	Dict(:A=>true,  :B=>true)=>0.1,
# 						                                          	Dict(:A=>false, :B=>true)=>0.2,
# 						                                          	Dict(:A=>true,  :B=>false)=>0.3,
# 						                                          	Dict(:A=>false, :B=>false)=>0.4,
# 						                                          	   )
# 						                                         )
# 					            ),
# 					],
# 			[(:A,:C), (:B,:C)]
# 		)

# 	ordering = topological_sort_by_dfs(bn.dag)
# 	@test ordering == [1,2,3] || ordering == [2,1,3]
# 	@test sort!(names(bn)) == [:A, :B, :C]
# 	@test isempty(parents(bn, :A))
# 	@test isempty(parents(bn, :B))
# 	@test sort!(parents(bn, :C)) == [:A, :B]
# 	@test isvalid(bn)
# end

# let
# 	bn = BayesNet([:B, :S, :E, :D, :C]) # construct an edgeless network with five variables
#                                     # note that this defaults to binary variables with 50/50 Bernoulli CPDs
# 	add_edges!(bn, [(:B, :E), (:S, :E), (:E, :D), (:E, :C)]) # add edges, does not change CPDs

# 	set_CPD!(bn, :B, CPDs.Bernoulli(0.1))
# 	set_CPD!(bn, :S, CPDs.Bernoulli(0.5))
# 	set_CPD!(bn, :E, CPDs.Bernoulli([:B, :S], rand_bernoulli_dict(2)))
# 	set_CPD!(bn, :D, CPDs.Bernoulli([:E], rand_bernoulli_dict(1)))
# 	set_CPD!(bn, :C, CPDs.Bernoulli([:E], rand_bernoulli_dict(1)))

# 	table(bn, :D) # ensure that this doens't fail
# end