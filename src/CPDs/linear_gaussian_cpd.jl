"""
A linear Gaussian CPD, always returns a Normal

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct LinearGaussianCPD <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    a::Vector{Float64}
    b::Float64
    σ::Float64
end
LinearGaussianCPD(target::NodeName, μ::Float64, σ::Float64) = LinearGaussianCPD(target, NodeName[], Float64[], μ, σ)

name(cpd::LinearGaussianCPD) = cpd.target
parents(cpd::LinearGaussianCPD) = cpd.parents
nparams(cpd::LinearGaussianCPD) = length(cpd.a) + 2
function (cpd::LinearGaussianCPD)(a::Assignment)

    # compute A⋅v + b
    μ = cpd.b
    for (i, p) in enumerate(cpd.parents)
        μ += a[p]*cpd.a[i]
    end

    Normal(μ, cpd.σ)
end
(cpd::LinearGaussianCPD)() = (cpd)(Assignment()) # cpd()
(cpd::LinearGaussianCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function Distributions.fit(::Type{LinearGaussianCPD},
    data::DataFrame,
    target::NodeName;
    min_stdev::Float64=0.0, # an optional minimum on the standard deviation
    )

    # no parents

    arr = data[!,target]
    eltype(arr) <: Real || error("fit LinearGaussianCPD requrires target to be numeric")

    μ = convert(Float64, mean(arr))
    σ = convert(Float64, stdm(arr, μ))
    σ = max(σ, min_stdev)

    LinearGaussianCPD(target, NodeName[], Float64[], μ, σ)
end
function Distributions.fit(::Type{LinearGaussianCPD},
    data::DataFrame,
    target::NodeName,
    parents::NodeNames;
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
    X = Array{Float64}(undef, nrow(data), nparents+1)
    for (i,p) in enumerate(parents)
        arr = data[!,p]
    	for j in 1 : nrow(data)
            X[j,i] = convert(Float64, arr[j])
    	end
    end
    X[:,end] .= 1.0

    y = convert(Vector{Float64}, data[!,target])

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

    LinearGaussianCPD(target, parents, a, b, σ)
end
