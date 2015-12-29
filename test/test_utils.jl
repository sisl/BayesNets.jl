df1 = DataFrame(
	A = [false, true, false, true],
	B = [false, false, true, true],
	p = [0.75, 0.60, 0.25, 0.40]
	)

df2 = DataFrame(
	A = [false, true],
	p = [0.9, 0.1]
	)

df12 = df1 * df2
@test elementwise_isapprox(select(df12, Dict(:A=>false, :B=>false))[:p], [0.75*0.9])
@test elementwise_isapprox(select(df12, Dict(:A=>true,  :B=>false))[:p], [0.60*0.1])
@test elementwise_isapprox(select(df12, Dict(:A=>false, :B=>true))[:p], [0.25*0.9])
@test elementwise_isapprox(select(df12, Dict(:A=>true,  :B=>true))[:p], [0.40*0.1])
