let
	data = DataFrame(
	        A = [1, 2, 1, 2, 2, 1, 2],
	        B = [1, 1, 1, 3, 3, 2, 1],
	        C = [1, 2, 2, 1, 1, 2, 1],
	    )

	cpdA = CPDs.CategoricalCPD(:A, 2)
	@test !trained(cpdA)

	learn!(cpdA, CPD[], data)
	@test trained(cpdA)
	@test name(cpdA) == :A
	@test ncategories(cpdA) == 2

	distrA = pdf(cpdA, Assignment())
	@test ncategories(distrA) == 2
	@test isapprox(pdf(distrA, 1), 3/7)
	@test isapprox(pdf(distrA, 2), 4/7)

	cpdB = learn!(CPDs.CategoricalCPD(:B, 3), [cpdA], data)
	@test trained(cpdB)
	@test name(cpdB) == :B
	@test ncategories(cpdB) == 3

	distrB = pdf(cpdB, Dict{NodeName, Any}(:A=>1))
	@test ncategories(distrB) == 3
	@test isapprox(pdf(distrB, 1), 2/3)
	@test isapprox(pdf(distrB, 2), 1/3)
	@test isapprox(pdf(distrB, 3), 0/3)

	distrB = pdf(cpdB, Dict{NodeName, Any}(:A=>2))
	@test isapprox(pdf(distrB, 1), 2/4)
	@test isapprox(pdf(distrB, 2), 0/4)
	@test isapprox(pdf(distrB, 3), 2/4)

	cpdC = learn!(CPDs.CategoricalCPD(:C, 2), [cpdA, cpdB], data)
	@test trained(cpdC)
	@test name(cpdC) == :C
	@test ncategories(cpdC) == 2

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>1, :B=>1))
	@test ncategories(distrC) == 2
	@test isapprox(pdf(distrC, 1), 1/2)
	@test isapprox(pdf(distrC, 2), 1/2)

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>2, :B=>3))
	@test ncategories(distrC) == 2
	@test isapprox(pdf(distrC, 1), 2/2)
	@test isapprox(pdf(distrC, 2), 0/2)

	# test with alpha
	cpdC = learn!(CPDs.CategoricalCPD(:C, 2, 1.0), [cpdA, cpdB], data)
	@test trained(cpdC)
	@test name(cpdC) == :C
	@test ncategories(cpdC) == 2

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>1, :B=>1))
	@test ncategories(distrC) == 2
	@test isapprox(pdf(distrC, 1), 2/4)
	@test isapprox(pdf(distrC, 2), 2/4)

	distrC = pdf(cpdC, Dict{NodeName, Any}(:A=>2, :B=>3))
	@test ncategories(distrC) == 2
	@test isapprox(pdf(distrC, 1), 3/4)
	@test isapprox(pdf(distrC, 2), 1/4)
end