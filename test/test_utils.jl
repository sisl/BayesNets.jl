let
    @test BayesNets.strip_arg(:symbol) == :symbol

    @test BayesNets.paramcount(:symbol) == 1
    @test BayesNets.paramcount(true) == 1
    @test BayesNets.paramcount(false) == 1
    @test BayesNets.paramcount(1.0) == 1
    @test BayesNets.paramcount(Array(Float64, 3)) == 3
end
