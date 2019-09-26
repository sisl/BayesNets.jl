#
# Tests for Factors (all of them)
#

let
@test_throws ArgumentError Factor([:X, :X], [2, 3])
@test_throws DimensionMismatch Factor([:A, :X], [3])
@test_throws DimensionMismatch Factor([:Y, :A, :X], [2, 3])
end

let
ϕ = Factor([:X, :Y], [3, 3], 16)
@test all(ϕ.potential .== 16)
end

let
# doesn't freak out
ϕ = Factor([:X, :Y], [4, 3], nothing)
rand!(ϕ)
end

let
ϕ = Factor([:X, :Y, :Z], [2, 3, 4])

@test eltype(ϕ) == Float64
@test ndims(ϕ) == 3
@test size(ϕ, :X) == 2
@test size(ϕ) == (2, 3, 4)
@test size(ϕ, :Y, :X) == (3, 2)

push!(ϕ, :A, 5)

@test ndims(ϕ) == 4
@test size(ϕ) == (2, 3, 4, 5)
@test size(ϕ, :A, :X) == (5, 2)

permutedims!(ϕ, [4, 3, 1, 2])

@test names(ϕ) == [:A, :Z, :X, :Y]
@test size(ϕ) == (5, 4, 2, 3)
end

let
bn = rand_discrete_bn(10, 4)
name = :N5

ϕ = Factor(bn, name)
df = join(DataFrame(ϕ), table(bn, name).potential, on=names(ϕ))
diff = abs.(df[:p] - df[:potential])

@test all(diff .< 1E-13)
end

###############################################################################
#                   patterns
let
ϕ = Factor([:l1, :l2], [2, 3])

@test pattern(ϕ, :l1) == [1 2 1 2 1 2]'
@test ϕ[:l2] == [1 1 2 2 3 3]'
@test pattern(ϕ, [:l1, :l2]) == pattern(ϕ)
@test pattern(ϕ) == [1 1; 2 1; 1 2; 2 2; 1 3; 2 3]
end

###############################################################################
#                   normalize
let
ϕ = Factor([:a, :b], Float64[1 2; 3 4])
ϕ2 = LinearAlgebra.normalize(ϕ, p=1)

@test elementwise_isapprox(ϕ2.potential, [0.1 0.2; 0.3 0.4])
@test elementwise_isapprox(ϕ.potential, Float64[1 2; 3 4])

@test_throws ArgumentError LinearAlgebra.normalize(ϕ, :waldo)

LinearAlgebra.normalize!(ϕ, p=2)

@test elementwise_isapprox(ϕ.potential, [1/30 2/30; 1/10 4/30])
end

###############################################################################
#                   broadcast
let
ϕ = Factor([:X, :Y], Float64[1 2; 3 4; 5 6])

@test elementwise_isapprox(
        broadcast(*, ϕ, [:Y, :X], [[10, 0.1], 100.0]).potential,
        Float64[1000 20; 3000 40; 5000 60])

@test_throws ArgumentError broadcast(*, ϕ, [:X, :Z], [[10, 1, 0.1], [1, 2, 3]])

@test_throws ArgumentError broadcast(*, ϕ, [:Z, :X, :A], [2, [10, 1, 0.1], [1, 2, 3]])

@test_throws DimensionMismatch broadcast(*, ϕ, :X, [2016, 58.0])
end

let
ϕ = Factor([:X, :Y, :Z], [3, 2, 2])
ϕ.potential[:] = Float64[1, 2, 3, 2, 3, 4, 4, 6, 7, 8, 10, 16]

@test DataFrame(broadcast(+, broadcast(+, ϕ, :Z, [10, 0.1]), :X, 10.0)) ==
        DataFrame(broadcast(+, ϕ, [:X, :Z], [10.0, [10, 0.1]]))
end

###############################################################################
#                   reduce dims
let
ϕ = Factor([:X, :Y, :Z], [3, 2, 2])
ϕ.potential[:] = Float64[1, 2, 3, 2, 3, 4, 4, 6, 7, 8, 10, 16]

df_original = DataFrame(ϕ)

@test_throws ArgumentError reducedim!(*, ϕ, :waldo)
# make sure it didn't change ϕ
@test DataFrame(ϕ) == df_original

# squeeze does some weird stuff man ...
@test sum(broadcast(*, ϕ, :Z, 0.0), names(ϕ)).potential == dropdims([0.0], dims=1)

df = DataFrame(X = [1, 2, 3], potential = [123.0, 165.0, 237.0])

ϕ2 = broadcast(*, ϕ, :Z, [1, 10.0])
sum!(ϕ2, [:Y, :Z])

# ϕ didn't change
@test DataFrame(ϕ2) != df_original
@test DataFrame(ϕ2) == df

@test_throws ArgumentError sum(ϕ, [:Y, :K, :Z, :waldo])
end

###############################################################################
#                   indexing
let
ϕ = Factor([:X, :Y, :Z], [3, 2, 2])
ϕ.potential[:] = [1, 2, 3, 2, 3, 4, 4, 6, 7, 8, 10, 16]

df = DataFrame(ϕ)

@test_throws TypeError ϕ[Assignment(:Y => "waldo")]
@test_throws BoundsError ϕ[Assignment(:Y => 16)]

a = Assignment(:Y=> 2, :K => 16, :Z => 1)
ϕ[a].potential  == Float64[2, 3, 4]

ϕ[Assignment(:X => 2, :Y => 1, :Z => 2)] = 1600.0
@test ϕ.potential[2, 1, 2] == 1600.0
@test DataFrame(ϕ)[LinearIndices(size(ϕ))[2, 1, 2], :potential] == 1600.0

ϕ[Assignment(:X => 1, :Y => 2)] = 2016
@test ϕ.potential[1, 2, :] == Float64[2016, 2016]
end

###############################################################################
#                   joins
# definitely more tests needed

let
ϕ1 = Factor([:X, :C, :Y, :A, :Z], [3, 2, 3, 2, 3])
ϕ2 = Factor([:A, :B, :C], [2, 3, 2])
ϕ1.potential[:] = collect(1.0:length(ϕ1))
ϕ2.potential[:] = collect(1.0:length(ϕ2))

# why did I do this?
true_pot = [1.0,2.0,3.0,28.0,35.0,42.0,7.0,8.0,9.0,70.0,77.0,84.0,13.0,14.0,
         15.0,112.0,119.0,126.0,38.0,40.0,42.0,176.0,184.0,192.0,50.0,52.0,
         54.0,224.0,232.0,240.0,62.0,64.0,66.0,272.0,280.0,288.0,37.0,38.0,
         39.0,280.0,287.0,294.0,43.0,44.0,45.0,322.0,329.0,336.0,49.0,50.0,
         51.0,364.0,371.0,378.0,110.0,112.0,114.0,464.0,472.0,480.0,122.0,
         124.0,126.0,512.0,520.0,528.0,134.0,136.0,138.0,560.0,568.0,576.0,
         73.0,74.0,75.0,532.0,539.0,546.0,79.0,80.0,81.0,574.0,581.0,588.0,
         85.0,86.0,87.0,616.0,623.0,630.0,182.0,184.0,186.0,752.0,760.0,768.0,
         194.0,196.0,198.0,800.0,808.0,816.0,206.0,208.0,210.0,848.0,856.0,
         864.0,3.0,6.0,9.0,36.0,45.0,54.0,21.0,24.0,27.0,90.0,99.0,108.0,39.0,
         42.0,45.0,144.0,153.0,162.0,76.0,80.0,84.0,220.0,230.0,240.0,100.0,
         104.0,108.0,280.0,290.0,300.0,124.0,128.0,132.0,340.0,350.0,360.0,
         111.0,114.0,117.0,360.0,369.0,378.0,129.0,132.0,135.0,414.0,423.0,
         432.0,147.0,150.0,153.0,468.0,477.0,486.0,220.0,224.0,228.0,580.0,
         590.0,600.0,244.0,248.0,252.0,640.0,650.0,660.0,268.0,272.0,276.0,
         700.0,710.0,720.0,219.0,222.0,225.0,684.0,693.0,702.0,237.0,240.0,
         243.0,738.0,747.0,756.0,255.0,258.0,261.0,792.0,801.0,810.0,364.0,
         368.0,372.0,940.0,950.0,960.0,388.0,392.0,396.0,1000.0,1010.0,1020.0,
         412.0,416.0,420.0,1060.0,1070.0,1080.0,5.0,10.0,15.0,44.0,55.0,66.0,
         35.0,40.0,45.0,110.0,121.0,132.0,65.0,70.0,75.0,176.0,187.0,198.0,
         114.0,120.0,126.0,264.0,276.0,288.0,150.0,156.0,162.0,336.0,348.0,
         360.0,186.0,192.0,198.0,408.0,420.0,432.0,185.0,190.0,195.0,440.0,
         451.0,462.0,215.0,220.0,225.0,506.0,517.0,528.0,245.0,250.0,255.0,
         572.0,583.0,594.0,330.0,336.0,342.0,696.0,708.0,720.0,366.0,372.0,
         378.0,768.0,780.0,792.0,402.0,408.0,414.0,840.0,852.0,864.0,365.0,
         370.0,375.0,836.0,847.0,858.0,395.0,400.0,405.0,902.0,913.0,924.0,
         425.0,430.0,435.0,968.0,979.0,990.0,546.0,552.0,558.0,1128.0,1140.0,
         1152.0,582.0,588.0,594.0,1200.0,1212.0,1224.0,618.0,624.0,630.0,
         1272.0,1284.0,1296.0]

ϕ12 = ϕ1 * ϕ2
@test ϕ12.potential[:] == true_pot
end

let
# dimensions don't have the same lengths
ϕ1 = Factor([:X, :Y, :Z], [3, 3, 3])
ϕ2 = Factor([:A, :X, :Y], [2, 31, 3])

@test_throws DimensionMismatch ϕ1 * ϕ2
end

