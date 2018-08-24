f1 = DataFrame(
    A = [false, true, false, true],
    B = [false, false, true, true],
    p = [0.75, 0.60, 0.25, 0.40]
    ) |> Table

f2 = DataFrame(
    A = [false, true],
    p = [0.9, 0.1]
    ) |> Table

let
    # factor multiplication
    f12 = f1 * f2
    @test size(f12) == (4,3)
    @test elementwise_isapprox(partialsort(f12, Assignment(:A=>false, :B=>false))[:p], [0.75*0.9])
    @test elementwise_isapprox(partialsort(f12, Assignment(:A=>true,  :B=>false))[:p], [0.60*0.1])
    @test elementwise_isapprox(partialsort(f12, Assignment(:A=>false, :B=>true))[:p], [0.25*0.9])
    @test elementwise_isapprox(partialsort(f12, Assignment(:A=>true,  :B=>true))[:p], [0.40*0.1])
end

let
    # factor marginalization
    f1_sans_B = sumout(f1, :B)
    @test size(f1_sans_B) == (2,2)
    @test elementwise_isapprox(partialsort(f1_sans_B, Assignment(:A=>false))[:p], [0.75 + 0.25])
    @test elementwise_isapprox(partialsort(f1_sans_B, Assignment(:A=>true))[:p], [0.60 + 0.40])

    f1_sans_A = sumout(f1, :A)
    @test size(f1_sans_A) == (2,2)
    @test elementwise_isapprox(partialsort(f1_sans_A, Assignment(:B=>false))[:p], [0.75 + 0.60])
    @test elementwise_isapprox(partialsort(f1_sans_A, Assignment(:B=>true))[:p], [0.25 + 0.40])
end

let
    # factor normalization
    f3 = BayesNets.normalize!(Table(DataFrame(
        A = [false, true],
        p = [1.0, 3.0]
    )))

    @test elementwise_isapprox(f3[:p], [0.25, 0.75])
end

let
    # estimation
    table = fit(Table, DataFrame(
        A = [false, false, true, true, true]
    ))
    @test elementwise_isapprox(table[:p], [2/5, 3/5])

    # TODO: properly test this
    # estimate_convergence(df, Assignment(:A=>true))
end
