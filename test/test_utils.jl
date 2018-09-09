let
    @test BayesNets.strip_arg(:symbol) == :symbol

    @test BayesNets.CPDs.paramcount(:symbol) == 1
    @test BayesNets.CPDs.paramcount(true) == 1
    @test BayesNets.CPDs.paramcount(false) == 1
    @test BayesNets.CPDs.paramcount(1.0) == 1
    @test BayesNets.CPDs.paramcount(Array{Float64}(undef, 3)) == 3
end
