#=
A linear gaussian CPD

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

	P(x|parents(x)) = Normal(μ=a*parents(x) + b, σ)
=#

type LinearGaussianCPD <: CPD{Normal{Float64}}

	core::CPDCore{Normal{Float64}}

	# data only initialized if has parents
	a::Vector{Float64}
	b::Float64

	LinearGaussianCPD(core::CPDCore{Normal{Float64}}) = new(core)
	function LinearGaussianCPD(
		core::CPDCore{Normal{Float64}},
		a::Vector{Float64},
		b::Float64,
		)

		new(core, a, b)
	end
end

name(cpd::LinearGaussianCPD) = cpd.core.name
parents(cpd::LinearGaussianCPD) = cpd.core.parents
distribution(cpd::LinearGaussianCPD) = cpd.core.d

function condition!(cpd::LinearGaussianCPD, a::Assignment)
    if !parentless(cpd)

        # compute A⋅v + b
		μ = cpd.b
		for (i, p) in enumerate(cpd.core.parents)
			μ += a[p]*cpd.a[i]
		end

		cpd.core.d = Normal(μ, cpd.core.d.σ)
    end

    cpd.core.d
end

function Distributions.fit(::Type{LinearGaussianCPD}, data::DataFrame, target::NodeName;
    min_stdev::Float64=0.0, # an optional minimum on the standard deviation
    )

    # no parents

    arr = data[target]
    eltype(arr) <: Real || error("fit CategoricalCPD requrires target to be numeric")

    μ = mean(arr)
    σ = max(stdm(arr, μ), min_stdev)

    core = CPDCore{Normal{Float64}}(target, NodeName[], Normal(μ, σ))
    LinearGaussianCPD(core)
end
function Distributions.fit(::Type{LinearGaussianCPD}, data::DataFrame, target::NodeName, parents::Vector{NodeName};
	min_stdev::Float64=0.0, # an optional minimum on the standard deviation
	)

	if isempty(parents)
	    return fit(LinearGaussianCPD, data, target, min_stdev=min_stdev)
	end

	# ---------------------
    # pull parental dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.

	nparents = length(parents)
	X = Array(Float64, nrow(data), nparents+1)
	for (i,p) in enumerate(parents)
		arr = data[p]
		for j in 1 : nrow(data)
			X[j,i] = convert(Float64, arr[j])
		end
	end
	X[:,end] = 1.0

	y = convert(Vector{Float64}, data[target])

	# --------------------
	# solve the regression problem
	#   β = (XᵀX)⁻¹Xᵀy
	#
	#     X is the [nsamples × nparents+1] data matrix
	#     where the last column is 1.0
	#
	#     y is the [nsamples] vector of target values
	#
	# NOTE: this will fail if X is not full rank

	β = (X'*X)\(X'*y)

	a = β[1:nparents]
	b = β[end]
	σ = max(std(y), min_stdev)

	core = CPDCore(target, parents, Normal(NaN, σ))
	LinearGaussianCPD(core, a, b)
end