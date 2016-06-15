#=
A categorical distribution

    P(x|parents(x)) ∈ Categorical

    Assumes all parents are discrete (integers 1:Nᵢ)
=#

type CategoricalCPD <: CPD{Categorical}
    n_instantiations::Int # number of values in domain, 1:n_instantiations

    probabilities::Array{Float64} # n_instantiations × nparental_instantiations of parents
                                  # n_instantiations if no parents
    parental_assignments::Vector{Int} # preallocated array of parental assignments, in BN topological order
    parent_instantiation_counts::Tuple{Vararg{Int}} # list of integer instantiation counts, in BN topological order

    function CategoricalCPD(name::NodeName, n::Int, alpha::Float64=0.0)
        retval = new()

        retval.n_instantiations = n
        retval.alpha = alpha

        # other things are NOT instantiated yet

        retval
    end
end

# trained(cpd::CategoricalCPD) = isdefined(cpd, :probabilities)
# ncategories(cpd::CategoricalCPD) = cpd.n_instantiations

# function learn!{C<:CPD}(
#     cpd::CategoricalCPD,
#     target_name::NodeName,
#     data::DataFrame,
#     )

#     # no parents
#     probabilities = fill(cpd.alpha, cpd.n_instantiations)
#     for v in data[cpd_name]
#         probabilities[v] += 1
#     end
#     probabilities ./= nrow(data)

#     # NOTE: parental_assignments and parent_instantiation_counts
#     #       are NOT instantiated
#     cpd.probabilities = probabilities

#     cpd
# end
# function learn!{C<:CPD}(
#     cpd::CategoricalCPD,
#     target_name::NodeName,
#     parent_CPDs::AbstractVector{C},
#     parent_names::AbstractVector{NodeName},
#     data::DataFrame,
#     )

#     @assert(length(parent_CPDs) == length(parent_names))
#     @assert(reduce(&, map(p->(distribution(p) <: DiscreteUnivariateDistribution), parent_CPDs)),
#             "All parents must be discrete")

#     cpd_name = name(cpd)

#     if !isempty(parent_CPDs)

#         # ---------------------
#         # pull discrete dataset
#         # 1st row is all of the data for the 1st parent
#         # 2nd row is all of the data for the 2nd parent, etc.

#         nparents = length(parent_CPDs)
#         discrete_data = Array(Int, nparents, nrow(data))
#         for (i,p) in enumerate(parent_names)
#             arr = data[name(p)]
#             for j in 1 : nrow(data)
#                 discrete_data[i,j] = arr[j]
#             end
#         end

#         my_data = convert(Vector{Int}, data[cpd_name]) # for this variable only

#         # ---------------------
#         # calc parent_instantiation_counts

#         parent_instantiation_counts = Array(Int, nparents)
#         for (i,p) in enumerate(parent_CPDs)
#             parent_instantiation_counts[i] = ncategories(p)
#         end

#         # ---------------------
#         # pull sufficient statistics

#         q  = prod(parent_instantiation_counts)
#         stridevec = fill(1, nparents)
#         for k = 2:nparents
#             stridevec[k] = stridevec[k-1] * parent_instantiation_counts[k-1]
#         end
#         js = (discrete_data - 1)' * stridevec + 1

#         probs = full(sparse(my_data, vec(js), 1.0, cpd.n_instantiations, q)) # currently a set of counts

#         probs += cpd.alpha

#         for i in 1 : q
#             tot = sum(probs[:,i])
#             if tot > 0.0
#                 probs[:,i] ./= tot
#             else
#                 probs[:,i] = 1.0/cpd.n_instantiations
#             end
#         end

#         cpd.probabilities = probs
#         cpd.parental_assignments = Array(Int, nparents)
#         cpd.parent_instantiation_counts = tuple(parent_instantiation_counts...)
#     else
#         learn!(cpd, target_name, data)
#     end

#     cpd
# end
# function pdf(cpd::CategoricalCPD, a::Assignment, parent_names::AbstractVector{NodeName})

#     if !isempty(parent_names)
#         # pull the parental assignments
#         for (i,p) in enumerate(parent_names)
#             cpd.parental_assignments[i] = a[p]
#         end

#         # get the parental assignment index
#         j = sub2ind_vec(cpd.parent_instantiation_counts, cpd.parental_assignments)

#         # build the distribution
#         Categorical(cpd.probabilities[:,j]) # NOTE: slicing the array is a copy (when this code was written)
#     else
#         Categorical(copy(cpd.probabilities)) # NOTE: slicing the array is a copy (when this code was written)
#     end
# end

