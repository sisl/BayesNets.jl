#=
A linear gaussian CPD

    Is compatible with Normal{Float64}

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

	P(x|parents(x)) = Normal(μ=a*parents(x) + b, σ)
=#

type LinearGaussianCPD <: CPDForm
	a::Vector{Float64}
	b::Float64
end

function condition!{D<:Normal{Float64},C<:LinearGaussianCPD}(cpd::CPD{D,C}, a::Assignment)

    # compute A⋅v + b
	μ = cpd.form.b
	for (i, p) in enumerate(cpd.parents)
		μ += a[p]*cpd.form.a[i]
	end

    cpd.d = Normal(μ, cpd.d.σ)
    cpd.d
end


function Distributions.fit{D<:Normal{Float64},C<:LinearGaussianCPD}(
    ::Type{CPD{D,C}},
    data::DataFrame,
    target::NodeName;
    min_stdev::Float64=0.0, # an optional minimum on the standard deviation
    )

    # no parents

    arr = data[target]
    eltype(arr) <: Real || error("fit LinearGaussianCPD requrires target to be numeric")

    μ = convert(Float64, mean(arr))
    σ = convert(Float64, stdm(arr, μ))
    σ = max(σ, min_stdev)

    CPD(target, Normal(μ, σ), LinearGaussianCPD(Float64[], μ))
end
function Distributions.fit{D<:Normal{Float64},C<:LinearGaussianCPD}(
    ::Type{CPD{D,C}},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName};
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

    CPD(target, parents, Normal(NaN, σ), LinearGaussianCPD(a, b))
end