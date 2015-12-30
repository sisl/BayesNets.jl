
# rand
@test rand([1.0, 0.0]) == 1
@test rand([0.0, 1.0]) == 2

let
	# DiscreteFunctionCPD
	# TODO: test other constructors

	d = DiscreteFunctionCPD(
			[1,2],
			a -> begin a
				if a == Dict(:A => 1)
					return [0.2, 0.8]
				elseif a == Dict(:A => 2)
					return [0.3, 0.7]
				elseif a == Dict(:A => 3)
					return [1.0, 0.0]
				end
				error("should not be reached")
			end
		)

	@test probvec(d, Dict(:A => 1)) == [0.2, 0.8]
	@test probvec(d, Dict(:A => 2)) == [0.3, 0.7]
	@test probvec(d, Dict(:A => 3)) == [1.0, 0.0]

	@test isa(pdf(d, Dict(:A => 1)), Function)
	@test rand(d, Dict(:A => 3)) == 1

	dom = domain(d)
	@test isa(dom, DiscreteDomain)
	@test convert(Vector{Int}, sort(dom.elements)) == [1,2]
end

let
	# DiscreteDictCPD
	d = DiscreteDictCPD(
			[1,2], Dict(Dict(:A => 1) => [0.2, 0.8],
						Dict(:A => 2) => [0.3, 0.7],
						Dict(:A => 3) => [1.0, 0.0])
		)

	@test probvec(d, Dict(:A => 1)) == [0.2, 0.8]
	@test probvec(d, Dict(:A => 2)) == [0.3, 0.7]
	@test probvec(d, Dict(:A => 3)) == [1.0, 0.0]

	@test isa(pdf(d, Dict(:A => 1)), Function)
	@test rand(d, Dict(:A => 3)) == 1

	dom = domain(d)
	@test isa(dom, DiscreteDomain)
	@test convert(Vector{Int}, sort(dom.elements)) == [1,2]
end

let
	# DiscreteStaticCPD
	d = DiscreteStaticCPD(
			[1,2], [0.5,0.5]
		)

	@test probvec(d, Dict(:A => 1)) == [0.5, 0.5]
	@test probvec(d, Dict(:A => 2)) == [0.5, 0.5]
	@test probvec(d, Dict(:A => 3)) == [0.5, 0.5]

	@test isa(pdf(d, Dict(:A => 1)), Function)

	dom = domain(d)
	@test isa(dom, DiscreteDomain)
	@test convert(Vector{Int}, sort(dom.elements)) == [1,2]
end

let
	# BernoulliCPD
	d = BernoulliCPD()
	@test probvec(d, Dict(:A => 1)) == [0.5, 0.5]
	@test probvec(d, Dict(:A => 2)) == [0.5, 0.5]

	d = BernoulliCPD(0.4)
	@test probvec(d, Dict(:A => 1)) == [0.4, 0.6]
	@test probvec(d, Dict(:A => 2)) == [0.4, 0.6]

	d = BernoulliCPD(a -> begin a
			if a == Dict(:A => 1)
				return 0.2
			elseif a == Dict(:A => 2)
				return 0.3
			elseif a == Dict(:A => 3)
				return 1.0
			end
			error("should not be reached")
		end
		)
	@test probvec(d, Dict(:A => 1)) == [0.2, 0.8]
	@test probvec(d, Dict(:A => 2)) == [0.3, 0.7]
	@test probvec(d, Dict(:A => 3)) == [1.0, 0.0]

	# TODO: test the final constructor once we figure out how it works
	# d = BernoulliCPD(

	# 		Dict(Dict(:A => 1) => [0.2, 0.8],
	# 		     Dict(:A => 2) => [0.3, 0.7],
	# 		     Dict(:A => 3) => [1.0, 0.0])
	# 	)
	# @test probvec(d, Dict(:A => 1)) == [0.2, 0.8]
	# @test probvec(d, Dict(:A => 2)) == [0.43, 0.7]
	# @test probvec(d, Dict(:A => 3)) == [1.0, 0.0]
end