"""
A conditional linear Gaussian CPD, always returns a Normal{Float64}

    This is a combination of the CategoricalCPD and the LinearGaussianCPD.
    For a variable with N discrete parents and M continuous parents, it will construct
    a linear gaussian distribution for all M parents for each discrete instantiation.

                      { Normal(μ=a₁×continuous_parents(x) + b₁, σ₁) for discrete instantiation 1
	P(x|parents(x)) = { Normal(μ=a₂×continuous_parents(x) + b₂, σ₂) for discrete instantiation 2
                      { ...
"""
struct ConditionalLinearGaussianCPD <: CPD{Normal}
    target::NodeName
    parents::NodeNames # list of all parents

    parents_disc::NodeNames # list of discrete parents
    parental_ncategories::Vector{Int} # list of instantiation counts for each discrete parent
    linear_gaussians::Vector{LinearGaussianCPD} # set of linear gaussian CPDs over the continuous parents
end

name(cpd::ConditionalLinearGaussianCPD) = cpd.target
parents(cpd::ConditionalLinearGaussianCPD) = cpd.parents
nparams(cpd::ConditionalLinearGaussianCPD) = sum(d->nparams(d), cpd.linear_gaussians)
function (cpd::ConditionalLinearGaussianCPD)(a::Assignment)

    idx = 1
    if !isempty(cpd.parents_disc)

        # get the index in cpd.distributions
        N = length(cpd.parents_disc)
        idx = a[cpd.parents_disc[N]] - 1
        for i in N-1:-1:1
            idx = (a[cpd.parents_disc[i]] - 1 + cpd.parental_ncategories[i]*idx)
        end
        idx += 1
    end

    lingaussian = cpd.linear_gaussians[idx]
    lingaussian(a)
end
(cpd::ConditionalLinearGaussianCPD)() = (cpd)(Assignment()) # cpd()
(cpd::ConditionalLinearGaussianCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function Distributions.fit(::Type{ConditionalLinearGaussianCPD},
    data::DataFrame,
    target::NodeName;
    min_stdev::Float64=0.0, # an optional minimum on the standard deviation
    )

    # no parents

    arr = data[!,target]
    eltype(arr) <: Real || error("fit ConditionalLinearGaussianCPD requrires target to be numeric")

    lingaussian = fit(LinearGaussianCPD, data, target, min_stdev=min_stdev)

    ConditionalLinearGaussianCPD(target, NodeName[], NodeName[], Int[], [lingaussian])
end
function Distributions.fit(::Type{ConditionalLinearGaussianCPD},
    data::DataFrame,
    target::NodeName,
    parents::NodeNames;
    min_stdev::Float64=0.0, # an optional minimum on the standard deviation
    )

    if isempty(parents)
        return fit(ConditionalLinearGaussianCPD, data, target, min_stdev=min_stdev)
    end

    # ---------------------
    # identify discrete and continuous parents

    parents_disc = filter(p->eltype(data[!,p]) <: Int, parents)
    parents_cont = filter(p->eltype(data[!,p]) <: AbstractFloat, parents)

   # ---------------------
    # pull discrete dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.
    # calc parent_instantiation_counts

    nparents_disc = length(parents_disc)

    if nparents_disc != 0

        parental_ncategories = Array{Int}(undef, nparents_disc)
        dims = Array{UnitRange{Int64}}(undef, nparents_disc)
        for (i,p) in enumerate(parents_disc)
            parental_ncategories[i] = infer_number_of_instantiations(data[!,p])
            dims[i] = 1:parental_ncategories[i]
        end

        # ---------------------
        # fit linear gaussians

        linear_gaussians = Array{LinearGaussianCPD}(undef, prod(parental_ncategories))
        for (q, parent_instantiation) in enumerate(Iterators.product(dims...))
            indeces = Int[]
            for i in 1 : nrow(data)
                if all(j->data[i,parents_disc[j]]==parent_instantiation[j], 1:nparents_disc) # parental instantiation matches
                    push!(indeces, i)
                end
            end
            linear_gaussians[q] = fit(LinearGaussianCPD, data[indeces, :], target, parents_cont, min_stdev=min_stdev)
        end
        ConditionalLinearGaussianCPD(target, parents, parents_disc, parental_ncategories, linear_gaussians)

    else # no discrete parents
        lingaussian = fit(LinearGaussianCPD, data, target, parents, min_stdev=min_stdev)
        ConditionalLinearGaussianCPD(target, parents, NodeName[], Int[], [lingaussian])
    end
end
