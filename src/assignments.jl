#=
An assignment of the variables in a Bayesian Network is represented as a dictionary.
=#

"""
True if all common keys between the two assignments have the same value
TODO: make more efficient
TODO: this fails if there are no common assignments:
		@test !consistent(
		Dict(Dict(:A => true) => 0.4,
			 Dict(:A => false) => 0.6),
		Dict(Dict(:A => true,  :B => true)  => 0.2,
			 Dict(:A => false, :B => false) => 0.1,
			 Dict(:A => false, :B => true)  => 0.4,
			 Dict(:A => true,  :B => false) => 0.3)
		)
"""
function consistent(a::Assignment, b::Assignment)
    commonKeys = intersect(keys(a), keys(b))
    reduce(&, [a[k] == b[k] for k in commonKeys])
end