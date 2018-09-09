let
	@test BayesNets.ndgrid([2,1]) == [2,1]
	@test elementwise_isapprox(BayesNets.ndgrid([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
	@test BayesNets.ndgrid([1,2], [3,4]) == (
											[1 1
											 2 2],
											[3 4
											 3 4])

	# TODO: make this test more rigorous
	@test isa(BayesNets.ndgrid([1], [2], [3]), Tuple{Array{Int64,3},Array{Int64,3},Array{Int64,3}})

	# TODO: make this test more rigorous
	@test BayesNets.ndgrid_fill!(Array{Int}(undef, 2), [1,2], 1, 2) == [1,2]

	@test BayesNets.meshgrid([1,2]) == (
										[1 2
										 1 2],
										[1 1
										 2 2])
	@test BayesNets.meshgrid([1,2], [3,4]) == (
												[1 2
												 1 2],
												[3 3
												 4 4])


	# TODO: make this test more rigorous
	@test isa(BayesNets.meshgrid([1], [2], [3]), Tuple{Array{Int64,3},Array{Int64,3},Array{Int64,3}})
end