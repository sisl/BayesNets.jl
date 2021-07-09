"""
A categorical distribution, P(x|parents(x)) where all parents are discrete integers 1:N.

The ordering of `distributions` array follows the convention in Decision Making Under Uncertainty.
Suppose a variable has three discrete parents. The first parental instantiation
assigns all parents to their first bin. The second will assign the first
parent (as defined in `parents`) to its second bin and the other parents
to their first bin. The sequence continues until all parents are instantiated
to their last bins.

This is equivalent to:

X,Y,Z
1,1,1
2,1,1
1,2,1
2,2,1
1,1,2
...
"""
struct CategoricalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    # list of instantiation counts for each parent, in same order as parents
    parental_ncategories::Vector{Int}
    # a vector of distributions in DMU order
    distributions::Vector{D}
end
CategoricalCPD(target::NodeName, d::D) where {D<:Distribution} =
        CategoricalCPD(target, NodeName[], Int[], D[d])

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parents
nparams(cpd::CategoricalCPD) = sum(d->paramcount(params(d)), cpd.distributions)

function (cpd::CategoricalCPD)(a::Assignment=Assignment())
    if isempty(cpd.parents)
        return first(cpd.distributions)
    else
        sub = [a[p] for p in cpd.parents]
        shape = ntuple(i -> cpd.parental_ncategories[i],
                length(cpd.parental_ncategories))
        ind = LinearIndices(shape)[sub...]
        return cpd.distributions[ind]
    end
end
(cpd::CategoricalCPD)(pair::Pair{NodeName}...) =
        (cpd)(Assignment(pair)) # cpd(:A=>1)

"""
    Distributions.ncategories(cpd::CategoricalCPD)

Return the number of categories for a cpd.
"""
Distributions.ncategories(cpd::CategoricalCPD) =
    ncategories(first(cpd.distributions))

function Distributions.fit(::Type{CategoricalCPD{D}},
                           data::DataFrame,
                           target::NodeName,
                           ) where {D}

    # no parents

    d = convert(D, fit(D, data[!,target]))
    CategoricalCPD(target, NodeName[], Int[], D[d])
end
function Distributions.fit(::Type{CategoricalCPD{D}},
                           data::DataFrame,
                           target::NodeName,
                           parents::NodeNames,
                           ) where {D}

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
    parental_ncategories = map!(p->infer_number_of_instantiations(data[!,p]), Array{Int}(undef, length(parents)), parents)
    dims = [1:parental_ncategories[i] for i in 1:nparents]
    distributions = Array{D}(undef, prod(parental_ncategories))
    for (q, parent_instantiation) in enumerate(Iterators.product(dims...))
        arr = Array{eltype(data[!,target])}(undef, 0)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                push!(arr, data[i, target])
            end
        end
        distributions[q] = fit(D, arr)
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end

#####

const DiscreteCPD = CategoricalCPD{Categorical{Float64,Vector{Float64}}}

DiscreteCPD(target::NodeName, prob::AbstractVector{T}) where {T<:Real} = CategoricalCPD(target, Categorical(prob ./ sum(prob)))

function Distributions.fit(::Type{DiscreteCPD},
    data::DataFrame,
    target::NodeName;
    ncategories::Int = infer_number_of_instantiations(data[!,target]),
    )

    d = convert(Categorical{Float64,Vector{Float64}}, fit_mle(Categorical, ncategories, data[!,target]))
    CategoricalCPD(target, NodeName[], Int[], Categorical{Float64,Vector{Float64}}[d])
end
function Distributions.fit(::Type{DiscreteCPD},
    data::DataFrame,
    target::NodeName,
    parents::NodeNames;
    parental_ncategories::Vector{Int} = map!(p->infer_number_of_instantiations(data[!,p]), Array{Int}(undef, length(parents)), parents),
    target_ncategories::Int = infer_number_of_instantiations(data[!,target]),
    )

    # with parents

    if isempty(parents)
        return fit(DiscreteCPD, data, target, ncategories=target_ncategories)
    end

    nparents = length(parents)
    dims = [1:parental_ncategories[i] for i in 1:nparents]
    distributions = Array{Categorical{Float64,Vector{Float64}}}(undef, prod(parental_ncategories))
    arr = Array{eltype(data[!,target])}(undef, 0)
    for (q, parent_instantiation) in enumerate(Iterators.product(dims...))
        empty!(arr)
        for i in 1 : nrow(data)
            if all(j->data[i,parents[j]]==parent_instantiation[j], 1:nparents) # parental instantiation matches
                push!(arr, data[i, target])
            end
        end
        if !isempty(arr)
            distributions[q] = fit_mle(Categorical, target_ncategories, arr)
        else
            distributions[q] = Categorical(target_ncategories)
        end
    end

    CategoricalCPD(target, parents, parental_ncategories, distributions)
end

