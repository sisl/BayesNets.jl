let
	bn = DiscreteBayesNet()
	push!(bn, rand_cpd(bn, 2, :a))
	push!(bn, rand_cpd(bn, 3, :b, [:a], uniform_dirichlet_prior=0.01))
end

let
	bn = DiscreteBayesNet()
	push!(bn, DiscreteCPD(:a, [0.4,0.6]))
	push!(bn, DiscreteCPD(:b, [:a], [2], [Categorical([0.5,0.5]),Categorical([0.2,0.8])]))

	T = table(bn, :a)
	@test T == DataFrame(a=[1,2], p=[0.4,0.6])

	T = table(bn, :b)
	@test T == DataFrame(a=[1,2,1,2], b=[1,1,2,2], p=[0.5,0.2,0.5,0.8])

	data = DataFrame(a=[1,1,1,1,2,2,2,2],
		             b=[1,2,1,2,1,1,1,2])
	T = count(bn, :a, data)
	@test T == DataFrame(a=[1,2], count=[4,4]) 

	T = count(bn, :b, data)
	@test nrow(T) == 4
	@test T[findfirst(i->T[i,:a] == 1 && T[i,:b] == 1, 1:4), :count] == 2
	@test T[findfirst(i->T[i,:a] == 2 && T[i,:b] == 1, 1:4), :count] == 3
	@test T[findfirst(i->T[i,:a] == 1 && T[i,:b] == 2, 1:4), :count] == 2
	@test T[findfirst(i->T[i,:a] == 2 && T[i,:b] == 2, 1:4), :count] == 1

	n = length(bn)
	parent_list = Array{Vector{Int}}(undef, n)
	bincounts = Array{Int}(undef, n)
	datamat = convert(Matrix{Int}, data)'

	for (i,cpd) in enumerate(bn.cpds)
		parent_list[i] = inneighbors(bn.dag, i)
		bincounts[i] = infer_number_of_instantiations(convert(Vector{Int}, data[i]))
	end

	@test isapprox(bayesian_score_component(1, parent_list[1], bincounts, datamat, UniformPrior()), -6.445719819385579)
	@test isapprox(bayesian_score_component(1, parent_list[1], bincounts, datamat, UniformPrior(2.0)), -6.13556489108174)
	@test isapprox(bayesian_score_component(1, parent_list[1], bincounts, datamat, BDeuPrior()), -6.841859646909766)
	@test isapprox(bayesian_score_component(1, parent_list[1], bincounts, datamat, BDeuPrior(2.0)), -6.445719819385579)
	@test isapprox(bayesian_score_component(2, parent_list[2], bincounts, datamat, UniformPrior()), -6.396929655216146)

	cache = ScoreComponentCache(data)
	@test isapprox(bayesian_score_component(1, parent_list[1], bincounts, datamat, UniformPrior()), -6.445719819385579)

	@test isapprox(bayesian_score(parent_list, bincounts, datamat, UniformPrior()), -6.445719819385579 + -6.396929655216146)
	@test isapprox(bayesian_score(bn, data), -6.445719819385579 + -6.396929655216146)
	@test isapprox(bayesian_score(bn, data, UniformPrior()), -6.445719819385579 + -6.396929655216146)

	score_components = bayesian_score_components(parent_list, bincounts, datamat, UniformPrior())
	@test isapprox(score_components[1], -6.445719819385579)
	@test isapprox(score_components[2], -6.396929655216146)

	score_components = bayesian_score_components(parent_list, bincounts, datamat, UniformPrior(), cache)
	@test isapprox(score_components[1], -6.445719819385579)
	@test isapprox(score_components[2], -6.396929655216146)

	score_components = bayesian_score_components(bn, data, UniformPrior())
	@test isapprox(score_components[1], -6.445719819385579)
	@test isapprox(score_components[2], -6.396929655216146)

    bs_from_structure = bayesian_score(bn.dag, Symbol[name(cpd) for cpd in bn.cpds], data)
    @test isapprox(bs_from_structure, -6.445719819385579 + -6.396929655216146)
    bs_from_structure = bayesian_score(bn.dag, Symbol[name(cpd) for cpd in bn.cpds], data, 2*ones(Int,nv(bn.dag)), UniformPrior())
    @test isapprox(bs_from_structure, -6.445719819385579 + -6.396929655216146)
end

let
	data = DataFrame(a=[1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,   2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2],
		             b=[1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,   1,1,1,1,2,2,2,2, 2,2,2,2,2,2,2,2],
		             c=[1,1,1,1,1,1,2,2, 2,2,2,2,2,2,1,1,   1,1,2,2,1,1,2,2, 1,1,1,1,1,1,1,1])

	cache = ScoreComponentCache(data)
	params = GreedyHillClimbing(cache, max_n_parents=3, prior=UniformPrior(2.0))

	@test params.max_n_parents == 3
	@test params.prior == UniformPrior(2.0)

	params = GreedyHillClimbing(cache)
	bn = fit(DiscreteBayesNet, data, params)
	@test length(bn) == ncol(data)
	@test isapprox(pdf(get(bn, :c), :c=>1), 0.61765, atol=1e-4)
	@test isapprox(pdf(get(bn, :b), :b=>1), 0.38235, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>1, :c=>1), 0.7, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>2, :c=>1), 0.21428, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>1, :c=>2), 0.5, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>2, :c=>2), 0.7, atol=1e-4)

end


# let
# 	file_data = readtable("schoolgrades.csv")
# 	cache = ScoreComponentCache(file_data)
# 	params = GreedyHillClimbing(cache, max_n_parents=3, prior=UniformPrior(2.0))
#
# 	println("Greedy Hill Climbing")
# 	params = GreedyHillClimbing(cache)
# 	bn = fit(DiscreteBayesNet, file_data, params)
#
# 	bs_from_structure = bayesian_score(bn.dag, Symbol[name(cpd) for cpd in bn.cpds], file_data)
# 	println(bs_from_structure)
#
# 	println("Greedy Thick Thinning")
# 	params = GreedyThickThinning(cache)
# 	bn = fit(DiscreteBayesNet, file_data, params)
# 	bs_from_structure = bayesian_score(bn.dag, Symbol[name(cpd) for cpd in bn.cpds], file_data)
# 	println(bs_from_structure)
# end

let
	data = DataFrame(a=[1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,   2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2],
		             b=[1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,   1,1,1,1,2,2,2,2, 2,2,2,2,2,2,2,2],
		             c=[1,1,1,1,1,1,2,2, 2,2,2,2,2,2,1,1,   1,1,2,2,1,1,2,2, 1,1,1,1,1,1,1,1])

	cache = ScoreComponentCache(data)
	params = ScanGreedyHillClimbing(cache, max_n_parents=3, max_depth=1, prior=UniformPrior(2.0))
	@test params.max_n_parents == 3
	@test params.prior == UniformPrior(2.0)

	params = ScanGreedyHillClimbing(cache)
	bn = fit(DiscreteBayesNet, data, params)
	@test length(bn) == ncol(data)
	@test isapprox(pdf(get(bn, :c), :c=>1), 0.61765, atol=1e-4)
	@test isapprox(pdf(get(bn, :b), :b=>1), 0.38235, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>1, :c=>1), 0.7, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>2, :c=>1), 0.21428, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>1, :c=>2), 0.5, atol=1e-4)
	@test isapprox(pdf(get(bn, :a), :a=>1, :b=>2, :c=>2), 0.7, atol=1e-4)
end

