using BayesNets
using PyPlot

srand(12345)
# TODO consider implementing slice sampling

function compute_distribution_from_samples(samples::DataFrame, bn::DiscreteBayesNet, shape::Array{Int64, 1}, is_weighted::Bool)
	result = zeros(shape...)
	num_samples = size(samples)[1]
	assert(num_samples > 0)

	as_array = convert(Array{Int64}, samples[:, names(bn)])

	for row in 1:num_samples
		add = 1
		if is_weighted
			add = samples[row, :p]
		end
		result[as_array[row, :]...] += add
	end
	if ~ is_weighted
		result /= num_samples
	end
	return result
end

function compute_true_distribution(bn::DiscreteBayesNet)
	a = Assignment(Dict(name => 1 for name in names(bn)))
        shape = [support(get(bn, name)(a))[end] for name in names(bn)]
	result = zeros(shape...)
	shape = size(result)
	for i in 1:prod(shape)
		assignment_index = ind2sub(shape, i)
		assignment = Assignment(Dict(ind_name[2] => convert(Integer, assignment_index[ind_name[1]]) for ind_name in enumerate(names(bn))))
		result[assignment_index...] = pdf(bn, assignment)
	end
	return result
end

function remove_conditioned_variables(distribution::Array, bn::DiscreteBayesNet, consistent_with::Assignment)
	if length(consistent_with) == 0
		return distribution
	end

        a = Assignment(Dict(name => 1 for name in names(bn)))
	slice_array = [haskey(consistent_with, name) ? convert(Int64, consistent_with[name]) : support(get(bn, name)(a)) for name in names(bn)]
	result = view(distribution, slice_array...)
	result /= sum(result)
	if any(isnan(result))
		println("Created NaN when removing conditioned variables")
		println("Evidence: ")
		println(consistent_with)
		for name in keys(consistent_with)
			column = findfirst(names(bn), name)
			print("Checking var: ")
			print(name)
			assert( all(distribution[:, column].== convert(Int64, consistent_with[name])) )
		end
	end
	return result
end

function compute_errors(samples::DataFrame, bn::DiscreteBayesNet, is_weighted::Bool, true_distribution::Array, consistent_with::Assignment)
	a = Assignment(Dict(name => 1 for name in names(bn)))
	shape = [support(get(bn, name)(a))[end] for name in names(bn)]
	samples_distribution = compute_distribution_from_samples(samples, bn, shape, is_weighted)
	true_distribution = remove_conditioned_variables(true_distribution, bn, consistent_with)
	samples_distribution = remove_conditioned_variables(samples_distribution, bn, consistent_with)

	l1_distance = sum(abs(true_distribution - samples_distribution))
	samples_distribution = samples_distribution[true_distribution.!= 0.0]
	true_distribution = true_distribution[true_distribution.!= 0.0]
	if any(samples_distribution.== 0.0)
		KL_div = Inf
	else
		KL_div = sum(true_distribution.*( log(true_distribution) - log(samples_distribution) ))
	end
	return l1_distance, KL_div
end

# Discrete compare function
# TODO also compare against rejection sampling
function compare_discrete(bn::DiscreteBayesNet, name::String, consistent_with::Assignment, 
                              burn_in::Integer, thinning::Integer, use_time::Bool = true)
	# sample_size_comparisons = [4000, 7000, 10000, 20000, 35000, 50000, 75000, 100000, 200000, 500000, 1000000] # , 2000000, 5000000, 10000000]
        sample_size_comparisons = [50 * i for i in 1:100]
	rand_table_weighted(bn, nsamples=sample_size_comparisons[1], consistent_with=consistent_with) # The first time takes long, let function compile
	println("Running...")
	println(name)
	results = zeros(length(sample_size_comparisons), 7)
        results_index = 1
	println("Computing true distribution...")
	true_distribution = compute_true_distribution(bn)
        # println(true_distribution)
	println("Done.")
	for sample_size in sample_size_comparisons
		print("Sample size: ")
		println(sample_size)
                rtw_start_time = now()
		rtw_samples = rand_table_weighted(bn, nsamples=sample_size, consistent_with=consistent_with)
		rtw_duration = convert(Integer, now() - rtw_start_time)
                rtw_errors = compute_errors(rtw_samples, bn, true, true_distribution, consistent_with)

		rtw_duration = max(1, rtw_duration)
                time_limit = Nullable{Integer}()
                gibbs_sample_size = sample_size
                if (use_time)
                    time_limit = Nullable{Integer}(rtw_duration)
                    gibbs_sample_size = sample_size * 10
                end
                gibbs_start_time = now()	
		gibbs_samples = gibbs_sample(bn, gibbs_sample_size, burn_in; sample_skip=thinning,
                         consistent_with=consistent_with, time_limit=time_limit, 
                         error_if_time_out=false)
                gibbs_duration = convert(Integer, now() - gibbs_start_time)
                print("Gibbs samples: ")
                println(size(gibbs_samples)[1])
                print("Gibbs time (ms): ")
                println(gibbs_duration)
		gibbs_errors = (-1, -1)
		if size(gibbs_samples)[1] > 0
			gibbs_errors = compute_errors(gibbs_samples, bn, false, true_distribution, consistent_with)
		end

                results[results_index, 1] = rtw_duration
		results[results_index, 2] = sample_size
		results[results_index, 3] = size(gibbs_samples)[1]
		results[results_index, 4] = rtw_errors[1]
		results[results_index, 5] = gibbs_errors[1]
		results[results_index, 6] = rtw_errors[2]
		results[results_index, 7] = gibbs_errors[2]
		results_index = results_index + 1
	end

	for row in 1:size(results)[1]
		println(results[row, :])
	end

	# Consider making this a scatter plot
	# TODO average this over multiple attempts	
	# TODO record the actual time Gibbs sampling takes, don't use the time limit
        valid_Gibbs_indicies = [i for i in 1:size(results)[1] if results[i, 5] != -1]
        if use_time
            println("Plotting...")
            plot(results[:, 1], results[:, 4], label="Likelihood L1")
            plot(results[valid_Gibbs_indicies, 1], results[valid_Gibbs_indicies, 5], label="Gibbs L1")
        
            xlabel("time (ms)")
            title(name)
            legend()
            println("Saving Plot...")
            savefig(name)
    	    clf()
        end
        scatter(results[:, 2], results[:, 4], label="Likelihood L1")
        scatter(results[valid_Gibbs_indicies, 3], results[valid_Gibbs_indicies, 5], c="g", marker="x", label="Gibbs L1")
        plot([0, sample_size_comparisons[end]], [0, 0], "black")

        xlabel("# samples")
        title(name)
        legend()
        println("Saving Plot...")
        savefig(join([name, " samples"]))
        clf()

        if ~ (any(results[:, 6].== Inf) || any(results[:, 7].== Inf))
                valid_Gibbs_indicies = [i for i in 1:size(results)[1] if results[i, 7] != -1]
		if use_time
			println("Plotting KL divergence")
	                plot(results[:, 1], results[:, 6], label="Likelihood KL")
	                plot(results[valid_Gibbs_indicies, 1], results[valid_Gibbs_indicies, 7], label="Gibbs KL")
	
		        xlabel("time (ms)")
	        	title(name)
		        legend()
	        	println("Saving Plot...")
			savefig(join([name, " KL"]))
			clf()
		end
		scatter(results[:, 2], results[:, 6], label="Likelihood L1")
	        scatter(results[valid_Gibbs_indicies, 3], results[valid_Gibbs_indicies, 7], c="g", marker="x", label="Gibbs L1")
	        plot([0, sample_size_comparisons[end]], [0, 0], "black")
	
		xlabel("# samples")
	        title(name)
		legend()
	        println("Saving Plot...")
		savefig(join([name, " samples", " KL"]))
	        clf()
        end

end

# Continuous compare function
function compare_continuous(bn::BayesNet, name::String)

end

# One Discrete univariate distribution (one discrete variable)
println("Building univariate discrete test case")
d_bn = DiscreteBayesNet()
push!(d_bn, DiscreteCPD(:a, [0.001, 0.099, 0.6, 0.25, 0.05]))
#compare_discrete(d_bn, "Univariate Discrete", Assignment(), 50, 0)

# Discrete only
println("Building multivariate discrete test case")
d_bn = DiscreteBayesNet()
push!(d_bn, DiscreteCPD(:a, [0.1,0.9]))
push!(d_bn, DiscreteCPD(:b, [0.1,0.9]))
push!(d_bn, CategoricalCPD{Categorical{Float64}}(:c, [:a, :b], [2,2],
                [Categorical([1.0, 0.0]), Categorical([0.2, 0.8]), Categorical([0.1, 0.9]), Categorical([0.3, 0.7])]))
push!(d_bn, CategoricalCPD{Categorical{Float64}}(:d, [:c], [2,],
                [Categorical([0.99, 0.01]), Categorical([0.2, 0.8])]))
#compare_discrete(d_bn, "Multivariate Discrete", Assignment(), 100, 0)

# Discrete with unlikely events
println("Building multivariate discrete with unlikely events conditioned test case")
d_bn = DiscreteBayesNet()
push!(d_bn, DiscreteCPD(:a, [0.1,0.9]))
push!(d_bn, DiscreteCPD(:b, [0.1,0.9]))
push!(d_bn, CategoricalCPD{Categorical{Float64}}(:c, [:a, :b], [2,2],
                [Categorical([1.0, 0.0]), Categorical([0.2, 0.8]), Categorical([0.1, 0.9]), Categorical([0.3, 0.7])]))
push!(d_bn, CategoricalCPD{Categorical{Float64}}(:d, [:c], [2,],
                [Categorical([0.99, 0.01]), Categorical([0.2, 0.8])]))
#compare_discrete(d_bn, "Multivariate Discrete Conditioned", Assignment(:d => 1), 600, 0)

# Discrete complex with Conditioned

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
# compare_discrete(bn, "Complex Discrete Conditioned", Assignment(:E => 3, :G => 2, :H => 1, :I => 4), 10000, 0)

# Example from the book, see page 38
bn = DiscreteBayesNet()
push!(bn, DiscreteCPD(:C, [0.001,0.999]))
push!(bn, CategoricalCPD{Categorical{Float64}}(:D, [:C], [2,],
                [Categorical([0.001, 0.999]), Categorical([0.999, 0.001])]))
# compare_discrete(bn, "Rare Events", Assignment(:D => 2), 200, 0, false)

# One continuous distribution

# One continuous distribution that is multimodal (Mixture of Gaussians)

# From the paper page 9 part C

function mean_and_stddev_error(samples::DataFrame, bn::BayesNet, target_mu::Array{Float64, 1}, 
		target_sigma::Array{Float64, 2}, is_weighted::Bool, consistent_with::Assignment)
        num_samples = size(samples)[1]
        assert(num_samples > 0)
        _names = [name for name in names(bn) if !haskey(consistent_with, name)]
        num_vars = length(_names)

        empirical_mean = zeros(size(target_mu)...)
        empirical_cov = zeros(size(target_sigma)...)

        for row in 1:num_samples
                weight = 1.0/num_samples
                if is_weighted
                        weight = Float64(samples[row, :p])
                end
                row_as_vec = squeeze(convert(Array{Float64}, samples[row, _names]), 1)
                empirical_mean += row_as_vec * weight
        end

        for row in 1:num_samples
                weight = 1.0/ (num_samples - 1)
                if is_weighted
                        weight = samples[row, :p]
                end
                row_as_vec = squeeze(convert(Array{Float64}, samples[row, _names]), 1)
                mean_shifted_vec = reshape(row_as_vec - empirical_mean, (num_vars,1))
                empirical_cov += *(mean_shifted_vec , mean_shifted_vec.') * weight
        end

        mean_MSE = sum((empirical_mean - target_mu).^2)
        cov_MSE = sum((empirical_cov - target_sigma).^2)
        
        return mean_MSE + cov_MSE, mean_MSE, cov_MSE, empirical_mean, empirical_cov
end

function estimate_mean_and_stddev(bn::BayesNet, burn_in::Int, thinning::Int,
                                  name::String, target_mu::Array{Float64, 1}, target_sigma::Array{Float64, 2};
                                  sample_size_comparisons::Array{Int, 1}=[50 * i for i in 1:100],
                                  consistent_with::Assignment = Assignment())
        println("Running...")
        println(name)

        results = zeros(length(sample_size_comparisons), 5)
        results_index = 1

        for sample_size in sample_size_comparisons
                print("Sample size: ")
                println(sample_size)
                rtw_samples = rand_table_weighted(bn, nsamples=sample_size, consistent_with=consistent_with)
                rtw_error = mean_and_stddev_error(rtw_samples, bn, target_mu, target_sigma, true, consistent_with)

                gibbs_samples = gibbs_sample(bn, sample_size, burn_in; sample_skip=thinning,
                         consistent_with=consistent_with)
                gibbs_error = mean_and_stddev_error(gibbs_samples, bn, target_mu, target_sigma, false, consistent_with)

                results[results_index, 1] = sample_size
                results[results_index, 2] = rtw_error[1]
                results[results_index, 3] = gibbs_error[1]
                results[results_index, 4] = rtw_error[2]
                results[results_index, 5] = gibbs_error[2]
                results_index = results_index + 1
        end

        for row in 1:size(results)[1]
                println(results[row, :])
        end

        # TODO average this over multiple attempts
        scatter(results[:, 1], results[:, 2], c="blue", label="Likelihood MSE")
        scatter(results[:, 1], results[:, 3], c="green", marker="x", label="Gibbs MSE")
        scatter(results[:, 1], results[:, 4], c="black", label="Likelihood MSE (mean only)")
        scatter(results[:, 1], results[:, 5], c="red", marker="x", label="Gibbs MSE (mean only)")
        plot([0, sample_size_comparisons[end]], [0, 0], "black")

        xlabel("# samples")
        title(name)
        legend()
        println("Saving Plot...")
        savefig(join([name, " samples"]))
        clf()

end

bn = BayesNet()
mu = [0.0, 0.0]
sig1 = sqrt(1.08)
sig2 = sqrt(0.31)
rho = (0.54/sig1)/sig2
sigma = [[1.08 0.54]; [0.54 0.31]]

#=
push!(bn, LinearGaussianCPD(:x1, NodeName[], Float64[], mu[1], sigma[1,1]))
push!(bn, LinearGaussianCPD(:x2, NodeName[:x1], 
       Float64[sigma[1,2] / sigma[1,1]],
       mu[2] - sigma[1,2] / sigma[1,1] * mu[1],
       sigma[2,2] - sigma[1,2]*sigma[1,2]/sigma[1,1]))
=#

push!(bn, StaticCPD(:x1, Normal(mu[1], sig1)))
push!(bn, LinearGaussianCPD(:x2, NodeName[:x1],
     Float64[rho * sig2 / sig1],
     mu[2] - mu[1] * rho * sig2 / sig1,
     sqrt(sig2*sig2 * (1 - rho*rho))
     ))

assert(names(bn) == [:x1, :x2])
burn_in = 600
thinning = 0
estimate_mean_and_stddev(bn, burn_in, thinning, "Multivariate Gaussian", mu, sigma)

x2_value = 5.0 * (sig2 / sig1 / rho)
target_mean = (rho * sig1 / sig2) * x2_value
target_std = sqrt(sig1 * sig1 * (1 - rho*rho))
# x2_value = 1.5
#target_mu = [mu[1] + (x2_value -  mu[2]) * sigma[1,2] / sigma[2,2]]
#target_sigma = ones(1,1) * (sigma[1,1] - sigma[1,2]*sigma[1,2]/sigma[2,2])
target_mu = [target_mean]
target_sigma = ones(1,1) * target_std
println(target_mu)
println(target_sigma)

#estimate_mean_and_stddev(bn, burn_in, thinning, "Conditional Multivariate Gaussian", 
#				target_mu, target_sigma, consistent_with=Assignment(:x2 => x2_value))

# Hybrid - Two Gaussians

# Hybrid with unlikely events

# Hybrid (Complex)



