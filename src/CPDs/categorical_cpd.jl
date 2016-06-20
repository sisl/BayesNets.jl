"""
A categorical distribution, P(x|parents(x)) where all parents are discrete integers 1:Nᵢ
"""
type CategoricalCPD{D} <: CPD{D}

    target::NodeName
    parents::Vector{NodeName}

    parental_ncategories::Vector{Int} # list of instantiation counts for each parent, in same order as parents
    distributions::Vector{D}
end
CategoricalCPD(target::NodeName, d::Distribution) = CategoricalCPD(target, NodeName[], Int[], D[d])

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parents


function Base.call(cpd::CategoricalCPD, a::Assignment)

    idx = 1
    if !isempty(cpd.parents)

        # get the index in cpd.distributions

        N = length(cpd.parents)
        idx = a[cpd.parents[N]] - 1
        for i in N-1:-1:1
            idx = (a[cpd.parents[i]] - 1 + cpd.parental_ncategories[i]*idx)
        end
        idx += 1
    end

    cpd.distributions[idx]
end

function Distributions.fit{D}(::Type{CategoricalCPD{D}},
    data::DataFrame,
    target::NodeName,
    )

    # no parents

    d = fit(D, data[target])
    CategoricalCPD(target, NodeName[], Int[], D[d])
end
function Distributions.fit{D}(::Type{CategoricalCPD{D}},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName},
    )

    # with parents

    if isempty(parents)
        return fit(CategoricalCPD{D}, data, target)
    end

    # ---------------------
    # pull discrete dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.
    # calc parent_instantiation_counts

    nparents = length(parents)
    parental_ncategories = Array(Int, nparents)
    dims = Array(UnitRange{Int64}, nparents)
    for (i,p) in enumerate(parents)
        parental_ncategories[i] = infer_number_of_instantiations(data[p])
        dims[i] = 1:parental_ncategories[i]
    end

    # ---------------------
    # fit distributions

    distributions = Array(D, prod(parental_ncategories))
    for (q, parent_instantiation) in enumerate(product(dims...))
        arr = Array(eltype(data[target]), 0)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                push!(arr, data[i, target])
            end
        end
        distributions[q] = fit(D, arr)
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end

# For CategoricalCPD{Categorical} we want to ensure that all distributions fit with the corrent number of categories
function Distributions.fit(::Type{CategoricalCPD{Categorical}},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName},
    )

    # with parents

    if isempty(parents)
        return fit(CategoricalCPD{Categorical}, data, target)
    end

    # ---------------------
    # pull discrete dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.
    # calc parent_instantiation_counts

    nparents = length(parents)
    parental_ncategories = Array(Int, nparents)
    dims = Array(UnitRange{Int64}, nparents)
    for (i,p) in enumerate(parents)
        parental_ncategories[i] = infer_number_of_instantiations(data[p])
        dims[i] = 1:parental_ncategories[i]
    end

    # ---------------------
    # fit distributions

    k = infer_number_of_instantiations(data[target])
    distributions = Array(Categorical, prod(parental_ncategories))
    for (q, parent_instantiation) in enumerate(product(dims...))
        arr = Array(eltype(data[target]), 0)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                push!(arr, data[i, target])
            end
        end
        distributions[q] = fit_mle(Categorical, k, arr)
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end

