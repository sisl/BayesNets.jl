using BayesNets
using PyPlot

srand(123)

function get_samples(bn::BayesNet, target::Symbol, a::Assignment, thinning::Integer, nsamples::Integer)

	println("Starting sampling...")

	gss = GibbsSamplerState(bn)
	var_distribution = get(bn, target)(a)
	samples = zeros(nsamples)

        for sample_iter in 1:nsamples
		sample_posterior_continuous!(gss, target, a, var_distribution; MH_iterations=thinning+1)
                samples[sample_iter] = a[target]

		if mod(sample_iter, nsamples/5) == 0
			print(sample_iter)
			print(" out of ")
			println(nsamples)
			println(a[target])
		end
        end

	return samples
end

function build_histogram(bn::BayesNet, target::Symbol, a::Assignment, thinning::Integer, 
               nsamples::Integer, target_distribution::Distribution, name::String; burn_in::Integer=0)

	println(name)

	samples = get_samples(bn, target, a, thinning, nsamples + burn_in)
        samples = samples[(burn_in+1):end, :]
	println("Plotting...")
	# weights = ones(size(samples)) / Float64(nsamples)
	n, bins, patches = plt[:hist](samples; bins=75, normed=true, facecolor="red", alpha=0.5)
        plot(bins, [pdf(target_distribution, bin) for bin in bins], "b--")
	title(name)
	xlabel("Variable Value")
	ylabel("PDF")
	savefig(join([name, ".png"]))
	clf()	

	lw_samples = rand_table_weighted(bn; nsamples = nsamples, 
		consistent_with = Assignment(name => a[name] for name in names(bn) if name != target))
	weights = convert(Array, lw_samples[:p])
        lw_samples = convert(Array, lw_samples[target])
        n, bins, patches = plt[:hist](lw_samples; bins=75, normed=true, facecolor="red", alpha=0.5)
	plot(bins, [pdf(target_distribution, bin) for bin in bins], "b--")
	name = join([name, " LW"])
        title(name)
        xlabel("Variable Value")
        ylabel("PDF")
        savefig(join([name, ".png"]))
        clf()

end

function single_var_hist(d::Distribution, name::String)

	known_stddev = BayesNet()
	dist = d
	push!(known_stddev, StaticCPD(:a, dist))
	known_target = :a
	known_target_distribution = dist
	known_a = Assignment(:a => 0.1)
	known_thinning = 0
	known_nsamples = 20000
	
	build_histogram(known_stddev, known_target, known_a, known_thinning,
	               known_nsamples, known_target_distribution, name)
	

end

# single_var_hist(Biweight(0.0, 1.0), "Biweight(0, 1)")
# single_var_hist(Chisq(1), "Chisq(1)")
# single_var_hist(InverseGaussian(1, 0.8), "InverseGaussian(1, 0.8)")

function two_variable_hist(sig1::Float64, sig2::Float64, correlation::Float64, name::String)
    bn = BayesNet() # unknown stddev (unknown to the MH algorithm)
    push!(bn, StaticCPD(:x1, Normal(0, sig1)))
    push!(bn, LinearGaussianCPD(:x2, NodeName[:x1],
       Float64[correlation * sig2 / sig1],
       0.0,
       sqrt(sig2*sig2 * (1 - correlation*correlation))
       ))

    x2_value = 5.0 * (sig2 / sig1 / correlation)
    target_mean = (correlation * sig1 / sig2) * x2_value
    target_std = sqrt(sig1 * sig1 * (1 - correlation*correlation))
    target = Normal( target_mean, target_std)
    print("Target Mean: ")
    print(target_mean)
    print(" Target std: ")
    println( target_std )
    print("x2_value: ")
    println(x2_value)
    a = Assignment(:x1 => 0.0, :x2 => x2_value)
    thinning = 1
    nsamples = 40000
    burn_in = Int(40000 / (thinning + 1))

    build_histogram(bn, :x1, a, thinning,
                       nsamples, target, name; burn_in=burn_in)

    mb = markov_blanket_cpds(bn, :x1)
    test_a = Assignment(:x1 => 3.0, :x2 => x2_value)
    test_a_out = exp(sum([logpdf(cpd, test_a) for cpd in mb]))
    test_a2 = Assignment(:x1 => 5.0, :x2 => x2_value)
    test_a2_out = exp(sum([logpdf(cpd, test_a2) for cpd in mb]))
    test_a3 = Assignment(:x1 => 7.0, :x2 => x2_value)
    test_a3_out = exp(sum([logpdf(cpd, test_a3) for cpd in mb]))
    println("x1 = 3, 5, 7 (out puts should be symmetric around 5")
    println("3")
    println(test_a_out)
    println(pdf(bn, test_a))
    println("5")
    println(test_a2_out)
    println(pdf(bn, test_a2))
    println("7")
    println(test_a3_out)
    println(pdf(bn, test_a3))
end

# two_variable_hist(1.0, 1.0, 0.4, "092 stddev") # posterior has mean 5 std 0.92
# two_variable_hist(1.0, 1.0, 0.8, "06 stddev") # posterior has mean 5 std 0.6
# two_variable_hist(1.0, 1.0, 0.99, "014 stddev") # posterior has mean 5 std 0.14
# two_variable_hist(1.0, 1.0, 0.999, "0045 stddev") # posterior has mean 5 std 0.045

# Is the conditional variance always less than the unconditional variance
# If it is, then we will never be in a case where the proposal has lower variance than the actual distribution
# Acutally this is not always true, TODO test with large stdddevs

