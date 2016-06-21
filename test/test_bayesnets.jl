let
	bn = BayesNet()
	@test length(bn) == 0

	push!(bn, StaticCPD(:a, Normal(0.0,1.0)))

	@test length(bn) == 1
	@test name(get(bn, :a)) == :a
	@test name(get(bn, 1)) == :a
	@test names(bn) == [:a]
	@test parents(bn, :a) == NodeName[]
	@test children(bn, :a) == NodeName[]
	@test pdf(bn, Assignment(:a=>0.0)) == pdf(Normal(0.0,1.0), 0.0)
	@test logpdf(bn, Assignment(:a=>0.0)) == logpdf(Normal(0.0,1.0), 0.0)
	@test isapprox(pdf(bn, DataFrame(a=[0.0, 1.0])), pdf(Normal(0.0,1.0), 0.0) * pdf(Normal(0.0,1.0), 1.0))
	@test isapprox(logpdf(bn, DataFrame(a=[0.0, 1.0])), logpdf(Normal(0.0,1.0), 0.0) + logpdf(Normal(0.0,1.0), 1.0))

	# b = 2a + 1
	push!(bn, LinearGaussianCPD(:b, [:a], [2.0], 1.0, 1.0))
	@test length(bn) == 2
	@test name(get(bn, :b)) == :b
	@test name(get(bn, 2)) == :b
	@test has_edge(bn, :a, :b)
	@test names(bn) == [:a, :b]
	@test parents(bn, :a) == NodeName[]
	@test children(bn, :a) == [:b]
	@test parents(bn, :b) == [:a]
	@test children(bn, :b) == NodeName[]
	@test pdf(bn, Assignment(:a=>1.0, :b=>2.0)) == pdf(Normal(0.0,1.0), 1.0) * pdf(Normal(3.0, 1.0), 2.0)
	@test logpdf(bn, Assignment(:a=>1.0, :b=>2.0)) == logpdf(Normal(0.0,1.0), 1.0) + logpdf(Normal(3.0, 1.0), 2.0)
end