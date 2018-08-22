let

        # A â~F~R C â~F~P B
        bn = BayesNet()
        push!(bn, StaticCPD(:a, Categorical([1.0,0.0])))
        push!(bn, StaticCPD(:b, Categorical([0.0,1.0])))
        push!(bn, CategoricalCPD{Bernoulli}(:c, [:a, :b], [2,2], [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(1.0), Bernoulli(0.4)]))

        t5 = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(), 
             variable_order=nothing, time_limit=nothing, 
             error_if_time_out=true, initial_sample=nothing)
        @test size(t5) == (5, 3)
        @test t5[:a] == [1,1,1,1,1]
        @test t5[:b] == [2,2,2,2,2]
        @test t5[:c] == [1,1,1,1,1]

        config = GibbsSampler(burn_in=100, thinning=5)
        t5 = rand(bn, config, 5)
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

        t6 = gibbs_sample(bn2, 5, 100; thinning=5, consistent_with=Assignment(:c=>1),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test t6[:a] == [1,1,1,1,1]
        @test t6[:b] == [1,1,1,1,1]
        @test t6[:c] == [1,1,1,1,1]

	config.evidence = Assignment(:c=>1)
	t6 = rand(bn2, config, 5)
	@test t6[:a] == [1,1,1,1,1]
        @test t6[:b] == [1,1,1,1,1]
        @test t6[:c] == [1,1,1,1,1]


        t7 = gibbs_sample(bn2, 5, 100; thinning=5, consistent_with=Assignment(:c=>2),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test t7[:a] == [2,2,2,2,2]
        @test t7[:b] == [1,1,1,1,1]
        @test t7[:c] == [2,2,2,2,2]

        rand()
	config.evidence = Assignment(:c=>2)
        t7 = rand(bn2, config, 5)
        @test t7[:a] == [2,2,2,2,2]
        @test t7[:b] == [1,1,1,1,1]
        @test t7[:c] == [2,2,2,2,2]

        bn3 = BayesNet()
        push!(bn3, StaticCPD(:a, Normal(2.5, 1.5)))
        push!(bn3, StaticCPD(:b, Categorical([0.3,0.3,0.4])))
        push!(bn3, LinearGaussianCPD(:c, [:a, :b], [0.5, 0.5], 1.0, 0.5))

        t8 = gibbs_sample(bn3, 5, 100; thinning=5, consistent_with=Assignment(),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test size(t8) == (5, 3)

        # unlikely c
        t9 = gibbs_sample(bn3, 5, 100; thinning=5, consistent_with=Assignment(:c=>1.0),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test size(t9) == (5, 3)

        # use optional parameters and border cases for other parameters
        t10 = gibbs_sample(bn3, 5, 0; thinning=0, consistent_with=Assignment(:c=>2.0, :b=>1),
             variable_order=(Vector{Symbol}([:c, :a, :b])), time_limit=1000000,
             error_if_time_out=false, initial_sample=Assignment(:a=>1, :b=>1, :c=>2.0))
        @test size(t10)[2] == 3 && size(t10)[1] <= 5

	config.burn_in = 0
	config.evidence = Assignment(:c=>2.0, :b=>1)
	config.variable_order = Vector{Symbol}([:c, :a, :b])
	config.time_limit = 1000000
	config.error_if_time_out = false
	config.initial_sample = Assignment(:a=>1, :b=>1, :c=>2.0)
	t10 = rand(bn3, config, 5)
        @test size(t10)[2] == 3 && size(t10)[1] <= 5

        # test early return from time limit
        t11 = gibbs_sample(bn3, 25000, 100; thinning=4, consistent_with=Assignment(),
             variable_order=nothing, time_limit= 1000,
             error_if_time_out=false, initial_sample=nothing)
        @test size(t10)[2] == 3 && size(t10)[1] <= 25000

	config = GibbsSampler(burn_in=100, thinning=4, time_limit = 1000, error_if_time_out=false)
	t11 = rand(bn3, config, 25000)
        @test size(t10)[2] == 3 && size(t10)[1] <= 25000

        d_bn = DiscreteBayesNet()
        push!(d_bn, DiscreteCPD(:a, [1.0,0.0]))
        push!(d_bn, DiscreteCPD(:b, [0.0,1.0]))
        push!(d_bn, CategoricalCPD{Categorical{Float64}}(:c, [:a, :b], [2,2], 
                        [Categorical([0.1, 0.9]), Categorical([0.2, 0.8]), Categorical([1.0, 0.0]), Categorical([0.4, 0.6])]))

	t12 = gibbs_sample(d_bn, 5, 100; thinning=5, consistent_with=Assignment(),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test size(t12) == (5, 3)
        @test t12[:a] == [1,1,1,1,1]
        @test t12[:b] == [2,2,2,2,2]
        @test t12[:c] == [1,1,1,1,1]


        t13 = gibbs_sample(d_bn, 5, 100; thinning=5, consistent_with=Assignment(:c=>1),
             variable_order=nothing, time_limit=nothing,
             error_if_time_out=true, initial_sample=nothing)
        @test size(t13) == (5, 3)
        @test t13[:a] == [1,1,1,1,1]
        @test t13[:b] == [2,2,2,2,2]
        @test t13[:c] == [1,1,1,1,1]

        # test bad parameters given to gibbs_sample
        pass = false
        try
            error_test = gibbs_sample(bn, 0, 100; thinning=5, consistent_with=Assignment(),
                 variable_order=nothing, time_limit=nothing,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn, 5, -1; thinning=5, consistent_with=Assignment(),
                 variable_order=nothing, time_limit=nothing,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn, 5, 100; thinning=-1, consistent_with=Assignment(),
                 variable_order=nothing, time_limit=nothing,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            v_order = Vector{Symbol}([:a, :c])
            error_test = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(),
                 variable_order=v_order, time_limit=nothing,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            v_order = Vector{Symbol}([:a, :c])
            config = GibbsSampler(burn_in=100, variable_order=v_order, thinning=5)
            error_test = rand(bn, config, 5)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            v_order = Vector{Symbol}([:a, :c, :b, :d])
            error_test = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(),
                 variable_order=v_order, time_limit=nothing,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(),
                 variable_order=nothing, time_limit=0,
                 error_if_time_out=true, initial_sample=nothing)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(),
                 variable_order=nothing, time_limit=nothing,
                 error_if_time_out=true, initial_sample=Assignment(:a => 1, :b => 1))
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            config = GibbsSampler(burn_in=100, thinning=5, initial_sample=Assignment(:a => 1, :b => 1))
            error_test = rand(bn, config, 5)
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn, 5, 100; thinning=5, consistent_with=Assignment(:c => 2),
                 variable_order=nothing, time_limit=nothing,
                 error_if_time_out=true, initial_sample=Assignment(:a => 1, :b => 1, :c => 1))
        catch e
            pass = true
        end
        @test pass

        pass = false
        try
            error_test = gibbs_sample(bn2, 5, 100; initial_sample=Assignment(:a => 1, :b => 1, :c => 3))
        catch e
            pass = true
        end
        @test pass
end
