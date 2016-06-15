#=
A categorical distribution

    P(x|parents(x)) ∈ Categorical

    Assumes the target and all parents are discrete integers 1:Nᵢ
=#

type CategoricalCPD <: CPD{Categorical}

    core::CPDCore{Categorical}

    # data only initialized if has parents
    parental_assignments::Vector{Int} # preallocated array of parental assignments, in BN topological order
    parent_instantiation_counts::Tuple{Vararg{Int}} # list of integer instantiation counts, in BN topological order
    probabilities::Matrix{Float64} # n_instantiations × nparental_instantiations of parents

    CategoricalCPD(core::CPDCore) = new(core)
    function CategoricalCPD(
        core::CPDCore,
        parental_assignments::Vector{Int},
        parent_instantiation_counts::Tuple{Vararg{Int}},
        probabilities::Matrix{Float64},
        )
        new(core, parental_assignments, parent_instantiation_counts, probabilities)
    end
end

name(cpd::CategoricalCPD) = cpd.core.name
parents(cpd::CategoricalCPD) = cpd.core.parents
distribution(cpd::CategoricalCPD) = cpd.core.d

function condition!(cpd::CategoricalCPD, a::Assignment)
    if !parentless(cpd)

        # pull the parental assignments
        for (i,p) in enumerate(cpd.core.parents)
            cpd.parental_assignments[i] = a[p]
        end

        # get the parental assignment index
        j = sub2ind_vec(cpd.parent_instantiation_counts, cpd.parental_assignments)

        # build the distribution
        p = cpd.core.d.p
        for i in 1 : length(p)
            p[i] = cpd.probabilities[i,j]
        end
    end

    cpd.core.d
end

function Distributions.fit(::Type{CategoricalCPD}, data::DataFrame, target::NodeName;
    dirichlet_prior::Float64=0.0, # prior counts
    )

    # no parents

    arr = data[target]
    eltype(arr) <: Int || error("fit CategoricalCPD requrires target to be an integer")

    n_instantiations = infer_number_of_instantiations(arr)

    probabilities = fill(dirichlet_prior, n_instantiations)
    for v in data[target]
        probabilities[v] += 1.0
    end
    probabilities ./= nrow(data)

    core = CPDCore(target, NodeName[], Categorical(probabilities))
    CategoricalCPD(core)
end
function Distributions.fit(::Type{CategoricalCPD}, data::DataFrame, target::NodeName, parents::Vector{NodeName};
    dirichlet_prior::Float64=0.0, # prior counts
    )

    # with parents

    if isempty(parents)
        return fit(CategoricalCPD, data, target, dirichlet_prior=dirichlet_prior)
    end

    # ---------------------
    # pull discrete dataset
    # 1st row is all of the data for the 1st parent
    # 2nd row is all of the data for the 2nd parent, etc.
    # calc parent_instantiation_counts

    nparents = length(parents)
    discrete_data = Array(Int, nparents, nrow(data))
    parent_instantiation_counts = Array(Int, nparents)
    for (i,p) in enumerate(parents)
        arr = data[p]
        parent_instantiation_counts[i] = infer_number_of_instantiations(arr)

        for j in 1 : nrow(data)
            discrete_data[i,j] = arr[j]
        end
    end

    # ---------------------
    # pull sufficient statistics

    q = prod(parent_instantiation_counts)
    stridevec = fill(1, nparents)
    for k = 2 : nparents
        stridevec[k] = stridevec[k-1] * parent_instantiation_counts[k-1]
    end
    js = (discrete_data - 1)' * stridevec + 1

    target_data = convert(Vector{Int}, data[target])
    n_instantiations = infer_number_of_instantiations(target_data)

    probs = full(sparse(target_data, vec(js), 1.0, n_instantiations, q)) # currently a set of counts
    probs = probs + dirichlet_prior

    for i in 1 : q
        tot = sum(probs[:,i])
        if tot > 0.0
            probs[:,i] ./= tot
        else
            probs[:,i] = 1.0/n_instantiations
        end
    end

    probabilities = probs
    parental_assignments = Array(Int, nparents)
    parent_instantiation_counts = tuple(parent_instantiation_counts...)

    core = CPDCore(target, parents, Categorical(n_instantiations))
    CategoricalCPD(core, parental_assignments, parent_instantiation_counts, probabilities)
end

