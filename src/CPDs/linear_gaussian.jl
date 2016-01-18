#=
A linear gaussian CPD

	P(x|parents(x)) = Normal(μ=a*parents(x) + b, σ)
=#

type LinearGaussianCPD <: CPD{Normal}

	a::Vector{Float64}
	b::Float64
	σ::Float64

	function LinearGaussianCPD(name::NodeName)

		retval = new()

		retval.name = name

		# other things are NOT instantiated yet

		retval
	end
end

trained(cpd::LinearGaussianCPD) = isdefined(cpd, :a)

function learn!{C<:CPD}(
	cpd::LinearGaussianCPD,
	target_name::NodeName,
	data::DataFrame,
	)

	# no parents

	μ = mean(data[target_name])
	σ = stdm(data[target_name], μ)

	cpd.a = Array(Float64, 0)
	cpd.b = μ
	cpd.σ = σ

	cpd
end
function learn!{C<:CPD}(
	cpd::LinearGaussianCPD,
	target_name::NodeName,
	parent_CPDs::AbstractVector{C},
	parent_names::AbstractVector{NodeName},
	data::DataFrame,
	)

	@assert(length(parent_CPDs) == length(parent_names))

	if !isempty(parents)

		# ---------------------
        # pull parental dataset
        # 1st row is all of the data for the 1st parent
        # 2nd row is all of the data for the 2nd parent, etc.

		nparents = length(parents)
		X = Array(Float64, nrow(data), nparents+1)
		for (i,p) in enumerate(parents)
			arr = data[parent_names[i]]
			for j in 1 : nrow(data)
				X[j,i] = convert(Float64, arr[j])
			end
		end
		X[:,end] = 1.0

		y = convert(Vector{Float64}, data[target_name])

		# --------------------
		# solve the regression problem
		#   β = (XᵀX)⁻¹Xᵀy
		#
		#     X is the [nsamples × nparents+1] data matrix
		#     where the last column is 1.0
		#
		#     y is the [nsamples] vector of target values
		#
		# NOTE that this will fail if X is not full rank

		β = (X'*X)\(X'*y)

		cpd.a = β[1:nparents]
		cpd.b = β[end]
		cpd.σ = std(y)
	else
		learn!(cpd, target_name, data)
	end

	cpd
end

function pdf(cpd::LinearGaussianCPD, a::Assignment, parent_names::AbstractVector{NodeName})

	nparents = length(parent_names)
	@assert(nparents == length(cpd.a))

	if nparents > 0

		# compute A⋅v + b
		μ = cpd.b
		for (i, sym) in enumerate(parent_names)
			μ += a[sym]*cpd.a[i]
		end

		Normal(μ, cpd.σ)
	else
		Normal(cpd.b, cpd.σ)
	end
end