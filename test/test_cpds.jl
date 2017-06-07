# StaticCPD
let
    df = DataFrame(a=randn(100))
    cpd = fit(StaticCPD{Normal}, df, :a)
    @test name(cpd) == :a
    @test parentless(cpd)
    @test parents(cpd) == NodeName[]
    @test disttype(cpd) <: Normal
    @test nparams(cpd) == 2
    @test isa(cpd(), Normal)
    @test pdf(cpd, :a=>0.5) > 0.2
    @test pdf(cpd, :a=>0.0) > pdf(cpd, :a=>0.5)
    @test logpdf(cpd, :a=>0.5) > log(0.2)
end

# CategoricalCPD
let

    # no parents
    let
        df = DataFrame(a=[1,2,1,2,3])
        cpd = fit(DiscreteCPD, df, :a)

        @test name(cpd) == :a
        @test parentless(cpd)
        @test nparams(cpd) == 3

        d = cpd(Assignment())
        @test isa(d, Categorical) && isa(d, disttype(cpd))
        @test pdf(d, 1) == 0.4
        @test pdf(d, 2) == 0.4
        @test pdf(d, 3) == 0.2

        df = DataFrame(a=randn(100))
        cpd = fit(CategoricalCPD{Normal}, df, :a)

        @test isa(cpd(), disttype(cpd))
        @test pdf(cpd, :a=>0.5) > 0.2
        @test pdf(cpd, :a=>0.0) > pdf(cpd, :a=>0.55)

        cpd = fit(DiscreteCPD, DataFrame(a=[1,2,1,2,2]), :a, ncategories=3)
        @test pdf(cpd, :a=>1) == 0.4
        @test pdf(cpd, :a=>2) == 0.6
        @test pdf(cpd, :a=>3) == 0.0
        @test nparams(cpd) == 3
    end

    # with parents
    let
        # Example with Categorical
        df = DataFrame(a=[1,2,1,2,3], b=[1,1,2,1,2])
        cpd = fit(DiscreteCPD, df, :b, [:a])

        @test name(cpd) == :b
        @test parents(cpd) == [:a]
        @test !parentless(cpd)
        @test nparams(cpd) == 6

        @test cpd(:a=>1).p == [0.5,0.5]
        @test cpd(:a=>2).p == [1.0,0.0]
        @test cpd(:a=>3).p == [0.0,1.0]

        cpd = fit(DiscreteCPD, df, :b, [:a], parental_ncategories=[3], target_ncategories=5)
        @test nparams(cpd) == 15


        # Example with Bernoulli and more than one parent
        df = DataFrame(a=[   1,    1,    1,    1,    2,    2,    2,    2],
                       b=[   1,    1,    2,    2,    1,    1,    2,    2],
                       c=[true, true,false,false, true,false,false, true])
        cpd = fit(CategoricalCPD{Bernoulli}, df, :c, [:a, :b])

        @test nparams(cpd) == 4

        @test isa(cpd(Assignment(:a=>1, :b=>1)), disttype(cpd))
        @test cpd(Assignment(:a=>1, :b=>1)).p == 1.0
        @test cpd(Assignment(:a=>1, :b=>2)).p == 0.0
        @test cpd(Assignment(:a=>2, :b=>1)).p == 0.5
        @test cpd(Assignment(:a=>2, :b=>2)).p == 0.5
    end
end

# Linear Gaussian
let

    # no parents
    let
        df = DataFrame(a=randn(100))
        cpd = fit(LinearGaussianCPD, df, :a, min_stdev=0.0)

        @test name(cpd) == :a
        @test parents(cpd) == NodeName[]
        @test parentless(cpd)
        @test disttype(cpd) <: Normal
        @test nparams(cpd) == 2
        @test pdf(cpd, :a=>0.5) > 0.2
        @test pdf(cpd, :a=>0.0) > pdf(cpd, :a=>0.55)
    end

    # with parents
    let
        a = randn(1000)
        b = randn(1000) .+ 2*a .+ 1

        df = DataFrame(a=a, b=b)
        cpd = fit(LinearGaussianCPD, df, :b, [:a])

        @test !parentless(cpd)
        @test parents(cpd) == [:a]
        @test nparams(cpd) == 3

        p = cpd(:a=>0.0)
        @test isapprox(p.μ, 1.0, atol=0.25)
        @test isapprox(p.σ, 2.0, atol=0.50)

        p = cpd(:a=>1.0)
        @test isapprox(p.μ, 3.0, atol=0.25)
        @test isapprox(p.σ, 2.0, atol=0.50)

        cpd = fit(LinearGaussianCPD, df, :b, [:a], min_stdev=10.0)
        @test cpd(:a=>1.0).σ == 10.0
    end
end

# ConditionalLinearGaussianCPD
let

    # no parents
    let
        df = DataFrame(a=randn(10))
        cpd = fit(ConditionalLinearGaussianCPD, df, :a)

        @test name(cpd) == :a
        @test parentless(cpd)
        @test nparams(cpd) == 2

        d = cpd()
        @test isa(d, Normal) && isa(d, disttype(cpd))
    end

    # with parents
    let
        df = DataFrame(a=[  1,   1,   1,   1,     2,   2,   2,   2],
                       b=[0.5, 1.0, 1.5, 1.0,   2.5, 3.0, 3.5, 3.0],
                       c=[0.55,1.05,1.53,1.02,  2.52,3.01,3.55,3.03])
        cpd = fit(ConditionalLinearGaussianCPD, df, :c, [:a, :b])

        @test name(cpd) == :c
        @test parents(cpd) == [:a, :b]
        @test !parentless(cpd)
        @test nparams(cpd) == 6

        d = cpd(Assignment(:a=>1, :b=>0.5))
        @test isapprox(d.μ, 0.5475, atol=0.001)
        @test isapprox(d.σ, 0.4003, atol=0.001)

        d = cpd(Assignment(:a=>2, :b=>3.0))
        @test isapprox(d.μ, 3.027, atol=0.001)
        @test isapprox(d.σ, 0.421, atol=0.001)
    end
end

# FunctionalCPD
let
    bn2 = BayesNet()
    push!(bn2, StaticCPD(:sighted, NamedCategorical([:bird, :plane, :superman], [0.40, 0.55, 0.05])))
    push!(bn2, FunctionalCPD{Bernoulli}(:happy, [:sighted], a->Bernoulli(a == :superman ? 0.95 : 0.2)))

    @test name(get(bn2, 2)) == :happy
    @test parents(get(bn2, 2)) == [:sighted]

    val = rand(bn2)
    @test in(val[:sighted], [:bird, :plane, :superman])
    @test in(val[:happy], [0,1])

    # named cat
    ncat = get(bn2, 1).d
    show(IOBuffer(), ncat)
    @test ncategories(ncat) == 3
    @test probs(ncat) == [0.40, 0.55, 0.05]
    @test params(ncat) == ([0.40, 0.55, 0.05],)
    @test pdf(ncat, :bird) == 0.4
    @test isapprox(logpdf(ncat, :bird), log(0.4))
    show(IOBuffer(), sampler(ncat))

    fcpd = FunctionalCPD{Normal}(:sad, a->Normal(0.0, 1.0))
    @test name(fcpd) == :sad
    @test isempty(parents(fcpd))
    @test fcpd(Assignment()) == Normal(0.0, 1.0)
end

# ParentFunctionalCPD
let
    a = StaticCPD(:a, Bernoulli(0.5))
    b = StaticCPD(:b, Bernoulli(0.6))
    p = [:a,:b]
    c = ParentFunctionalCPD{Bernoulli}(:c, p, (seq,par)->begin
                Bernoulli(mean(seq[k] for k in par))
            end
        )
    bn = BayesNet()
    push!(bn, a)
    push!(bn, b)
    push!(bn, c)
    @test mean(rand(bn, 20,:a=>0)[:c]) <= 0.6
    @test mean(rand(bn, 20, :a=>1, :b=>1)[:c]) == 1;
end
