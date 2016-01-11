let
	data = DataFrame(
			A = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], # N(0,1)
			B = [1.1, 2.1, -0.1, 1.9,  0.1, 1.8,  0.2, 2.1, -0.1], # N(A+1, σ)
			C = [2.0, 7.5, -3.5, 7.5, -3.5, 3.5, -7.5, 7.5, -3.5], # N(A + 2B, σ)
		)

	cpdA = CPDs.LinearGaussianCPD(:A)
	@test !trained(cpdA)

	learn!(cpdA, CPD[], data)
	@test trained(cpdA)
	@test name(cpdA) == :A

	distrA = pdf(cpdA, Assignment())
	@test isapprox(mean(distrA), 0.0)
	@test isapprox(std(distrA), 1.0)

	cpdB = learn!(CPDs.LinearGaussianCPD(:B), [cpdA], data)
	@test trained(cpdB)
	@test name(cpdB) == :B

	distrB = pdf(cpdB, Dict{NodeName, Any}(:A=>0.0))
	@test isapprox(mean(distrB), 1.01111111, atol=1e-8)
	@test isapprox(std(distrB), 0.98418043, atol=1e-8)

	distrB = pdf(cpdB, Dict{NodeName, Any}(:A=>1.0))
	@test isapprox(mean(distrB), 1.98611111, atol=1e-8)
	@test isapprox(std(distrB), 0.98418043, atol=1e-8)

	cpdC = learn!(CPDs.LinearGaussianCPD(:C), [cpdA, cpdB], data)
	@test trained(cpdC)
	@test name(cpdC) == :C

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>0.0, :B=>0.0))
	@test isapprox(mean(distrC), 0.48648648, atol=1e-8)
	@test isapprox(std(distrC), 5.77590781, atol=1e-8)

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>1.0, :B=>2.0))
	@test isapprox(mean(distrC), 6.61969111, atol=1e-8)
	@test isapprox(std(distrC), 5.77590781, atol=1e-8)
end