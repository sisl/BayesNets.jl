using BayesNets
using ProfileView

bn = DiscreteBayesNet()
push!(bn, DiscreteCPD(:A, [0.1,0.9]))
push!(bn, DiscreteCPD(:B, [0.5,0.5]))
push!(bn, rand_cpd(bn, 10, :C, [:A, :B]))
push!(bn, rand_cpd(bn, 4, :D, [:C]))
push!(bn, rand_cpd(bn, 4, :E, [:A, :C]))
push!(bn, rand_cpd(bn, 3, :F, [:E, :C]))
push!(bn, rand_cpd(bn, 4, :G, [:A, :B, :C, :D, :E, :F]))
push!(bn, rand_cpd(bn, 4, :H, [:A, :B, :F, :G]))
push!(bn, rand_cpd(bn, 6, :I, [:A, :B, :C, :F, :G]))
a = Assignment(:E => 3, :G => 2, :H => 1, :I => 4)

gibbs_sample(bn, 100, 100; sample_skip=2, consistent_with=Assignment(:E => 3, :G => 2, :H => 1, :I => 4), time_limit=Nullable{Integer}(9999999999))


@profile gibbs_sample(bn, 10000, 100; sample_skip=2, consistent_with=Assignment(:E => 3, :G => 2, :H => 1, :I => 4), time_limit=Nullable{Integer}(9999999999))

# Profile.print()

ProfileView.svgwrite("gibbs_profile_results.svg")

ProfileView.view()

@profile gibbs_sample(bn, 10000, 100; sample_skip=2, consistent_with=Assignment(:E => 3, :G => 2, :H => 1, :I => 4), time_limit=Nullable{Integer}(9999999999), max_cache_size=Nullable{Integer}(0))

# Profile.print()

ProfileView.svgwrite("gibbs_profile_results_no_caching.svg")

# ProfileView.view()

