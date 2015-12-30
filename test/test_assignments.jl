let
	# constistent
	@test consistent(
		Dict(Dict(:A => true) => 0.4,
			 Dict(:A => false) => 0.6),
		Dict(Dict(:A => true) => 0.4,
			 Dict(:A => false) => 0.6)
		)

	@test consistent(
		Dict(Dict(:A => true,  :B => true)  => 0.2,
			 Dict(:A => true,  :B => false) => 0.3,
			 Dict(:A => false, :B => true)  => 0.4,
			 Dict(:A => false, :B => false) => 0.1),
		Dict(Dict(:A => true,  :B => true)  => 0.2,
			 Dict(:A => false, :B => false) => 0.1,
			 Dict(:A => false, :B => true)  => 0.4,
			 Dict(:A => true,  :B => false) => 0.3)
		)

	@test !consistent(
		Dict(Dict(:A => true) => 0.4,
			 Dict(:A => false) => 0.6),
		Dict(Dict(:A => true) => 0.7,
			 Dict(:A => false) => 0.3)
		)

	# TODO: fix it failing if there are no common assignments
	# @test !consistent(
	# 	Dict(Dict(:A => true) => 0.4,
	# 		 Dict(:A => false) => 0.6),
	# 	Dict(Dict(:A => true,  :B => true)  => 0.2,
	# 		 Dict(:A => false, :B => false) => 0.1,
	# 		 Dict(:A => false, :B => true)  => 0.4,
	# 		 Dict(:A => true,  :B => false) => 0.3)
	# 	)
end