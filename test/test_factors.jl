#
# Tests for Factors (all of them)
#

let
@test_throws ArgumentError Factor([:X, :X], [2, 3])
@test_throws DimensionMismatch Factor([:A, :X], [3])
@test_throws DimensionMismatch Factor([:Y, :A, :X], [2, 3])
end

let
ft = Factor([:X, :Y], [3, 3], 16)
@test all(ft.probability .== 16)
end

let
# doesn't freak out
ft = Factor([:X, :Y], [4, 3], nothing)
rand!(ft)
end

let
ft = Factor([:X, :Y, :Z], [2, 3, 4])

@test eltype(ft) == Float64
@test ndims(ft) == 3
@test size(ft, :X) == 2
@test size(ft) == (2, 3, 4)
@test size(ft, :Y, :X) == (3, 2)

push!(ft, :A, 5)

@test ndims(ft) == 4
@test size(ft) == (2, 3, 4, 5)
@test size(ft, :A, :X) == (5, 2)

permutedims!(ft, [4, 3, 1, 2])

@test names(ft) == [:A, :Z, :X, :Y]
@test size(ft) == (5, 4, 2, 3)
end

let
bn = rand_discrete_bn(10, 4)
name = :N5

ft = Factor(bn, name)
df = join(DataFrame(ft), table(bn, name), on=names(ft))
diff = abs(df[:p] - df[:probability])

@test all(diff .< 1E-13)
end

###############################################################################
#                   patterns
let
ft = Factor([:l1, :l2], [2, 3])

@test pattern(ft, :l1) == [1 2 1 2 1 2]'
@test ft[:l2] == [1 1 2 2 3 3]'
@test pattern(ft, [:l1, :l2]) == pattern(ft)
@test pattern(ft) == [1 1; 2 1; 1 2; 2 2; 1 3; 2 3]
end

###############################################################################
#                   broadcast
let
ft = Factor([:X, :Y], Float64[1 2; 3 4; 5 6])

@test elementwise_isapprox(
        broadcast(*, ft, [:Y, :X], [[10, 0.1], 100]).probability,
        Float64[1000 20; 3000 40; 5000 60])
end

###############################################################################
#                   reduce dims
let
ft = Factor([:X, :Y, :Z], [3, 2, 2])
ft.probability[:] = Float64[1, 2, 3, 2, 3, 4, 4, 6, 7, 8, 10, 16]

df_original = DataFrame(ft)

@test_throws TypeError reducedim!(*, ft, "waldo")
# make sure it didn't change ft
@test DataFrame(ft) == df_original

@test DataFrame(broadcast(+, broadcast(+, ft, :Z, [10, 0.1]), :X, 10)) ==
        DataFrame(broadcast(+, ft, [:X, :Z], [10, [10, 0.1]])) 

# squeeze does some weird stuff man ...
@test sum(broadcast(*, ft, :Z, 0), names(ft)).probability == squeeze([0.0], 1)

    let
    df = DataFrame(X = [1, 2, 3], probability = [123.0, 165.0, 237.0])

    ft2 = broadcast(*, ft, :Z, [1, 10])
    sum!(ft2, [:Y, :Z])

    # ft didn't change
    @test DataFrame(ft2) != df_original
    @test DataFrame(ft2) == df
    end

    let
    df = DataFrame(X = [1, 2, 3], probability = [15, 21, 30])
    @test DataFrame(sum(ft, [:Y, :K, :Z, :waldo])) == df
    end
end

###############################################################################
#                   indexing
let
ft = Factor([:X, :Y, :Z], [3, 2, 2])
ft.probability[:] = [1, 2, 3, 2, 3, 4, 4, 6, 7, 8, 10, 16]

df = DataFrame(ft)

@test_throws TypeError ft[Assignment(:Y => "waldo")]
@test_throws BoundsError ft[Assignment(:Y => 16)]

a = Assignment(:Y=> 2, :K => 16, :Z => 1)
ft[a].probability  == Float64[2, 3, 4]

ft[Assignment(:X => 2, :Y => 1, :Z => 2)] = 1600.0
@test ft.probability[2, 1, 2] == 1600.0
@test DataFrame(ft)[sub2ind(ft, 2, 1, 2), :probability] == 1600.0

ft[Assignment(:X => 1, :Y => 2)] = 2016
@test ft.probability[1, 2, :] == Float64[2016, 2016]
end

# TODO test joins

