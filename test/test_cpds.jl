@test CPDs.sub2ind_vec((2,2), [1,1]) == 1
@test CPDs.sub2ind_vec((2,2), [2,1]) == 2
@test CPDs.sub2ind_vec((2,2), [1,2]) == 3
@test CPDs.sub2ind_vec((2,2), [2,2]) == 4
@test CPDs.sub2ind_vec((3,2,2), [1,1,1]) == 1
@test CPDs.sub2ind_vec((3,2,2), [2,1,1]) == 2
@test CPDs.sub2ind_vec((3,2,2), [3,1,1]) == 3
@test CPDs.sub2ind_vec((3,2,2), [1,2,1]) == 4
@test CPDs.sub2ind_vec((3,2,2), [2,2,1]) == 5
@test CPDs.sub2ind_vec((3,2,2), [1,1,2]) == 7

# StaticCPD
let
    df = DataFrame(a=randn(100))
    cpd = fit(StaticCPD{Normal}, df, :a)
    @test name(cpd) == :a
    @test isempty(parents(cpd))
    @test condition!(cpd, Dict{Symbol, Any}()) === distribution(cpd)
    @test pdf(cpd, Dict{Symbol, Any}(:a=>0.5)) > 0.2
    @test pdf(cpd, Dict{Symbol, Any}(:a=>0.0)) > pdf(cpd, Dict{Symbol, Any}(:a=>0.5))
end

# CategoricalCPD
let

    # no parents
    let
        df = DataFrame(a=[1,2,1,2,3])
        cpd = fit(CategoricalCPD, df, :a)

        @test name(cpd) == :a
        @test parentless(cpd)
        @test distribution(cpd) === condition!(cpd, Dict{Symbol, Any}())
        @test pdf(distribution(cpd), 1) == 0.4
        @test pdf(distribution(cpd), 2) == 0.4
        @test pdf(distribution(cpd), 3) == 0.2
    end

    # with parents
    let
        df = DataFrame(a=[1,2,1,2,3], b=[1,1,2,1,2])
        cpd = fit(CategoricalCPD, df, :b, [:a])

        @test !parentless(cpd)

        @test condition!(cpd, Dict(:a=>1)).p == [0.5,0.5]
        @test condition!(cpd, Dict(:a=>2)).p == [1.0,0.0]
        @test condition!(cpd, Dict(:a=>3)).p == [0.0,1.0]

        cpd = fit(CategoricalCPD, df, :b, [:a], dirichlet_prior=1.0)
        @test condition!(cpd, Dict(:a=>1)).p == [0.50,0.50]
        @test condition!(cpd, Dict(:a=>2)).p == [0.75,0.25]
        @test condition!(cpd, Dict(:a=>3)).p == [ 1/3, 2/3]
    end
end

# Linear Gaussian
let

    # no parents
    let
        df = DataFrame(a=randn(100))
        cpd = fit(LinearGaussianCPD, df, :a, min_stdev=0.0)

        @test name(cpd) == :a
        @test parentless(cpd)
        @test distribution(cpd) === condition!(cpd, Dict{Symbol, Any}())
        @test abs(distribution(cpd).σ - 1.0) < 0.2
        @test pdf(cpd, Dict{Symbol, Any}(:a=>0.5)) > 0.2
        @test pdf(cpd, Dict{Symbol, Any}(:a=>0.0)) > pdf(cpd, Dict{Symbol, Any}(:a=>0.5))
    end

    # with parents
    let
        a = randn(100)
        b = randn(100) .+ 2*a .+ 1

        df = DataFrame(a=a, b=b)
        cpd = fit(LinearGaussianCPD, df, :b, [:a])

        @test !parentless(cpd)

        p = condition!(cpd, Dict(:a=>0.0))
        @test isapprox(p.μ, 1.0, atol=0.25)
        @test isapprox(p.σ, 2.0, atol=0.25)

        p = condition!(cpd, Dict(:a=>1.0))
        @test isapprox(p.μ, 3.0, atol=0.25)
        @test isapprox(p.σ, 2.0, atol=0.25)

        cpd = fit(LinearGaussianCPD, df, :b, [:a], min_stdev=10.0)
        @test distribution(cpd).σ == 10.0
    end
end