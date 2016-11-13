let
	# A → C ← B
	bn = BayesNet()
	push!(bn, StaticCPD(:a, Categorical([1.0,0.0])))
	push!(bn, StaticCPD(:b, Categorical([0.0,1.0])))
	push!(bn, CategoricalCPD{Bernoulli}(:c, [:a, :b], [2,2], [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(1.0), Bernoulli(0.4)]))

	@test rand(bn) == Dict(:a=>1, :b=>2, :c=>1)

	t1 = rand(bn, 5)
	@test size(t1) == (5,3)
	@test t1[:a] == [1,1,1,1,1]
	@test t1[:b] == [2,2,2,2,2]

	t2 = rand(bn, 5, Assignment(:c=>1))
	@test size(t1) == (5,3)

	t3 = rand(bn, 5, :c=>1, :b=>2)
	@test size(t1) == (5,3)

	t4 = rand_table_weighted(bn; nsamples=5, consistent_with=Assignment(:c=>1))

        t5 = gibbs_sample(bn, 5, 100; sample_skip=5, consistent_with=Assignment(), 
             variable_order=Nullable{Vector{Symbol}}(), time_limit=Nullable{Integer}(), 
             error_if_time_out=true, inital_sample=Nullable{Assignment}())
        @test size(t5) == (5, 3)
        @test t5[:a] == [1,1,1,1,1]
        @test t5[:b] == [2,2,2,2,2]
        @test t5[:c] == [1,1,1,1,1]

        bn2 = BayesNet()
        push!(bn2, StaticCPD(:a, Categorical([0.5,0.5])))
        push!(bn2, StaticCPD(:b, Categorical([0.5,0.5])))
        push!(bn2, CategoricalCPD{Categorical}(:c, [:a, :b], [2,2], [
               Categorical([1.0, 0, 0, 0]), Categorical([0, 1.0, 0, 0]),
               Categorical([0, 0, 1.0, 0]), Categorical([0, 0, 0, 1.0])]))

        t6 = gibbs_sample(bn2, 5, 100; sample_skip=5, consistent_with=Assignment(:c=>1),
             variable_order=Nullable{Vector{Symbol}}(), time_limit=Nullable{Integer}(),
             error_if_time_out=true, inital_sample=Nullable{Assignment}())
        @test t6[:a] == [1,1,1,1,1]
        @test t6[:b] == [1,1,1,1,1]
        @test t6[:c] == [1,1,1,1,1]

        t7 = gibbs_sample(bn2, 5, 100; sample_skip=5, consistent_with=Assignment(:c=>2),
             variable_order=Nullable{Vector{Symbol}}(), time_limit=Nullable{Integer}(),
             error_if_time_out=true, inital_sample=Nullable{Assignment}())
        @test t7[:a] == [2,2,2,2,2]
        @test t7[:b] == [1,1,1,1,1]
        @test t7[:c] == [2,2,2,2,2]

        bn3 = BayesNet()
        push!(bn3, StaticCPD(:a, Normal(2.5, 1.5)))
        push!(bn3, StaticCPD(:b, Categorical([0.3,0.3,0.4])))
        push!(bn3, LinearGaussianCPD(:c, [:a, :b], [0.5, 0.5], 1.0, 0.5))

        t8 = gibbs_sample(bn3, 5, 100; sample_skip=5, consistent_with=Assignment(),
             variable_order=Nullable{Vector{Symbol}}(), time_limit=Nullable{Integer}(),
             error_if_time_out=true, inital_sample=Nullable{Assignment}())
        @test size(t8) == (5, 3)

        # unlikely c
        t9 = gibbs_sample(bn3, 5, 100; sample_skip=5, consistent_with=Assignment(:c=>1.0),
             variable_order=Nullable{Vector{Symbol}}(), time_limit=Nullable{Integer}(),
             error_if_time_out=true, inital_sample=Nullable{Assignment}())
        @test size(t9) == (5, 3)

        # TODO test bad parameters given to gibbs_sample

end
