#=
A linear gaussian CPD

	P(x|parents(x)) = Normal(μ=a*parents(x) + b, σ)
=#

type LinearGaussianCPD <: CPD{Normal}

	name::NodeName
	parent_names::Vector{NodeName} # ordering of the parents

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

function learn!{C<:CPD}(cpd::LinearGaussianCPD, parents::AbstractVector{C}, data::DataFrame)

	node_name = name(cpd)

	if !isempty(parents)

		# ---------------------
        # pull parental dataset
        # 1st row is all of the data for the 1st parent
        # 2nd row is all of the data for the 2nd parent, etc.

		nparents = length(parents)
		X = Array(Float64, nrow(data), nparents+1)
		for (i,p) in enumerate(parents)
			arr = data[name(p)]
			for j in 1 : nrow(data)
				X[j,i] = convert(Float64, arr[j])
			end
		end
		X[:,end] = 1.0

		y = convert(Vector{Float64}, data[node_name])

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
		μ = mean(data[node_name])
		σ = stdm(data[node_name], μ)

		cpd.a = Array(Float64, 0)
		cpd.b = μ
		cpd.σ = σ
	end

	cpd.parent_names = convert(Vector{NodeName}, map(p->name(p), parents))

	cpd
end
function pdf(cpd::LinearGaussianCPD, a::Assignment)

	if !isempty(cpd.parent_names)

		parent_values = Array(Float64, length(cpd.parent_names))
		for (i, sym) in enumerate(cpd.parent_names)
			parent_values[i] = a[sym]
		end

		μ = dot(cpd.a, parent_values) + cpd.b
		Normal(μ, cpd.σ)
	else
		Normal(cpd.b, cpd.σ)
	end
end