#=
A categorical distribution that assumes discrete parents
=#

type CategoricalCPD <: CPD{Categorical}
    n_instantiations::Int # number of values in domain, 1:n_instantiations

    parents::Vector{NodeName} # ordering of the parents

    probabilities::Array{Float64} # n_instantiations × nparental_instantiations of parents
                                  # n_instantiations if no parents
    parental_assignments::Vector{Int} # preallocated array of parental assignments
    parent_instantiation_counts::Tuple{Vararg{Int}} # list of integer instantiation counts

    function CategoricalCPD(n::Int)
        retval = new()
        retval.n_instantiations = n

        # other things are NOT instantiated now

        retval
    end
end

trained(cpd::CategoricalCPD) = isdefined(cpd, :probabilities)
Distributions.ncategories(cpd::CategoricalCPD) = cpd.n_instantiations

function learn!(cpd::CategoricalCPD, bn::BayesNet, name::NodeName, data::DataFrame, alpha::Float64=0.0)

    myparents = parents(bn, name)
    @assert(all(map(p->distribution(node(bn, p).cpd) <: DiscreteUnivariateDistribution, myparents)),
            "All parents must be discrete")

    if !isempty(myparents)

        # ---------------------
        # pull discrete dataset
        # 1st row is all of the data for the 1st parent
        # 2nd row is all of the data for the 2nd parent, etc.

        nparents = length(myparents)
        discrete_data = Array(Int, nparents, nrow(data))
        for (i,p) in enumerate(myparents)
            arr = data[p]
            for j in 1 : nrow(data)
                discrete_data[i,j] = arr[j]
            end
        end

        my_data = Array(Int, nrow(data)) # for this variable only
        arr = data[name]
        for j in 1 : nrow(data)
            my_data[j] = arr[j]
        end

        # ---------------------
        # calc parent_instantiation_counts

        parent_instantiation_counts = Array(Int, nparents)
        for (i,p) in enumerate(myparents)
            parent_instantiation_counts[i] = ncategories(node(bn,p).cpd)
        end

        # ---------------------
        # pull sufficient statistics

        q  = prod(parent_instantiation_counts)
        stridevec = fill(1, nparents)
        for k = 2:nparents
            stridevec[k] = stridevec[k-1] * parent_instantiation_counts[k-1]
        end
        js = (discrete_data - 1)' * stridevec + 1

        probs = full(sparse(my_data, vec(js), 1.0, cpd.n_instantiations, q)) # currently a set of counts

        probs += alpha

        for i in 1 : q
            tot = sum(probs[:,i])
            if tot > 0.0
                probs[:,i] ./= tot
            else
                probs[:,i] = 1.0/cpd.n_instantiations
            end
        end

        cpd.probabilities = probs
        cpd.parental_assignments = Array(Int, nparents)
        cpd.parent_instantiation_counts = tuple(parent_instantiation_counts...)
    else
        probabilities = fill(alpha, cpd.n_instantiations)
        for v in data[name]
            probabilities[v] += 1
        end
        probabilities ./= nrow(data)

        # NOTE: parental_assignments and parent_instantiation_counts
        #       are NOT instantiated
        cpd.probabilities = probabilities
    end

    cpd.parents = myparents

    cpd
end
function pdf(cpd::CategoricalCPD, a::Assignment)

    if !isempty(cpd.parents)
        # pull the parental assignments
        for (i,p) in enumerate(cpd.parents)
            cpd.parental_assignments[i] = a[p]
        end

        # get the parental assignment index
        j = sub2ind_vec(cpd.parent_instantiation_counts, cpd.parental_assignments)

        # build the distribution
        Categorical(cpd.probabilities[:,j]) # NOTE: slicing the array is a copy (at time of writing)
    else
        Categorical(copy(cpd.probabilities)) # NOTE: slicing the array is a copy (at time of writing)
    end
end

# df = DataFrame(
#         A = [1, 2, 1, 2, 2, 1, 2],
#         B = [1, 1, 1, 3, 3, 2, 1],
#         C = [1, 2, 2, 1, 1, 2, 1],
#     )
# cpd = CategoricalCPD(2)
# bn = BayesNet([:A, :B, :C])
# add_edge!(bn, :A, :C)
# add_edge!(bn, :B, :C)
# learn!(cpd, bn, :C, df)