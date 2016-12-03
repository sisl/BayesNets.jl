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

		if mod(sample_iter, nsamples/200) == 0
			print(sample_iter)
			print(" out of ")
			println(nsamples)
			println(a[target])
		end
        end

	return samples
end

function build_histogram(bn::BayesNet, target::Symbol, a::Assignment, thinning::Integer, 
               nsamples::Integer, target_distribution::Distribution, name::String)

	println(name)

	samples = get_samples(bn, target, a, thinning, nsamples)
	println("Plotting...")
	# weights = ones(size(samples)) / Float64(nsamples)
	n, bins, patches = plt[:hist](samples; bins=75, normed=true, facecolor="red", alpha=0.5)
        plot(bins, [pdf(target_distribution, bin) for bin in bins], "b--")
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

"""
immutable GMix{T<:Real} <: ContinuousUnivariateDistribution
    d::UnivariateGMM

    function GMix(p1::Vector{Float64}, p2::Vector{Float64}, p3::Categorical)
	new(UnivariateGMM(p1, p2, p3))
    end
end

var(d::GMix) = var(d.d)
pdf(d::GMix, x::Real) = pdf(d.d, x)

macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))

    paramdecl = D_has_constantbounds ? :(d::Union{$D, Type{$D}}) : :(d::$D)

    # overall
    esc(quote
        minimum($(paramdecl)) = $lb
        maximum($(paramdecl)) = $ub
    end)
end

@distr_support GMix -Inf Inf

dist = GMix{Float64}([-1.0, 1.0], [0.25, 0.25], Categorical([0.5, 0.5]))
single_var_hist(dist, "Normal(-1, 0.25) + Normal(1, 0.25)")
"""

function two_variable_hist(sig1::Float64, sig2::Float64, correlation::Float64, name::String)
    sigma = [[sig1*sig1 sig1*sig2*correlation]; [sig1*sig2*correlation sig2*sig2]]

    bn = BayesNet() # unknown stddev (unknown to the MH algorithm)
    push!(bn, StaticCPD(:x1, Normal(0, sigma[1, 1])))
    push!(bn, LinearGaussianCPD(:x2, NodeName[:x1],
       Float64[sigma[1,2] / sigma[1,1]],
       0.0,
       sigma[1,1] - sigma[1,2]*sigma[1,2]/sigma[2,2]))

    x2_value = 1.0
    target = Normal( (sigma[1,2] / sigma[2,2]) * x2_value, sigma[2,2] - sigma[1,2]*sigma[1,2]/sigma[1,1])
    a = Assignment(:x1 => 0.0, :x2 => x2_value)
    thinning = 0
    nsamples = 20000

    build_histogram(bn, :x1, a, thinning,
                       nsamples, target, name)
end


two_variable_hist(1.0, 0.1, 0.8, "Small stddev")
two_variable_hist(1.0, 10.0, 0.8, "Large stddev")

