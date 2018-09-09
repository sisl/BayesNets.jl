#=
A Bayesian Network (BN) represents a probability distribution
over a set of variables, P(x₁, x₂, ..., xₙ)
It leverages relations between variables in order to efficiently encode the joint distribution.
A BN is defined by a directed acyclic graph in which each node is a variable
and contains an associated conditional probability distribution P(xⱼ | parents(xⱼ))
=#

const DAG = DiGraph

function _build_dag_from_cpds(
	cpds::AbstractVector{T},
	name_to_index::Dict{NodeName, Int}
) where {T<:CPD}

	dag = DAG(length(cpds))
	for cpd in cpds
		j = name_to_index[name(cpd)]
		for p in parents(cpd)
			i = name_to_index[p]
			add_edge!(dag, i, j)
		end
	end
	dag
end
function _enforce_topological_order(
	dag::DAG,
	cpds::AbstractVector{T},
	name_to_index::Dict{NodeName, Int},
) where {T<:CPD}

	topo = topological_sort_by_dfs(dag)

	cpds2 = Array{T}(undef, length(cpds))
	name_to_index2 = Dict{NodeName, Int}()
	for (i,j) in enumerate(topo)
		# i is the new index
		# j is the old index

		cpd = cpds[j]
		name_to_index2[name(cpd)] = i
		cpds2[i] = cpd
	end

	dag2 = _build_dag_from_cpds(cpds2, name_to_index2)

	(dag2, cpds2, name_to_index2)
end

mutable struct BayesNet{T<:CPD} <: ProbabilisticGraphicalModel
	dag::DAG # nodes are in topological order
	cpds::Vector{T} # the CPDs associated with each node in the dag
	name_to_index::Dict{NodeName,Int} # NodeName → index in dag and cpds
end
BayesNet() = BayesNet(DAG(0), CPD[], Dict{NodeName, Int}())
BayesNet(::Type{T}) where {T <: CPD} = BayesNet(DAG(0), T[], Dict{NodeName, Int}())
function BayesNet(cpds::AbstractVector{T}) where {T <: CPD}

	name_to_index = Dict{NodeName, Int}()
	for (i, cpd) in enumerate(cpds)
		name_to_index[name(cpd)] = i
	end

	dag = _build_dag_from_cpds(cpds, name_to_index)

	!is_cyclic(dag) || error("BayesNet graph is non-acyclic!")

	(dag2, cpds2, name_to_index2) = _enforce_topological_order(dag, cpds, name_to_index)

	BayesNet(dag2, cpds2, name_to_index2)
end

Base.get(bn::BayesNet, i::Int) = bn.cpds[i]
Base.get(bn::BayesNet, name::NodeName) = bn.cpds[bn.name_to_index[name]]
Base.length(bn::BayesNet) = length(bn.cpds)

"""
Returns the ordered list of NodeNames
"""
function Base.names(bn::BayesNet)
	retval = Array{NodeName}(undef, length(bn))
	for (i,cpd) in enumerate(bn.cpds)
		retval[i] = name(cpd)
	end
	retval
end

"""
Returns the parents as a list of NodeNames
"""
CPDs.parents(bn::BayesNet, name::NodeName) = parents(get(bn, name))

"""
Returns the children as a list of NodeNames
"""
function children(bn::BayesNet, target::NodeName)
	i = bn.name_to_index[target]
	NodeName[name(bn.cpds[j]) for j in outneighbors(bn.dag, i)]
end

"""
Returns all neighbors as a list of NodeNames.
"""
function neighbors(bn::BayesNet, target::NodeName)
	i = bn.name_to_index[target]
	NodeName[name(bn.cpds[j]) for j in append!(inneighbors(bn.dag, i), outneighbors(bn.dag, i))]
end

"""
Returns all descendants as a list of NodeNames.
"""
dst(edge::Pair{Int,Int}) = edge[2] # LightGraphs used to return a Pair, now it returns a SimpleEdge
function descendants(bn::BayesNet, target::NodeName)
	retval = Set{Int}()
	for edge in edges(bfs_tree(bn.dag, bn.name_to_index[target]))
		push!(retval, dst(edge))
	end
	NodeName[name(bn.cpds[i]) for i in sort!(collect(retval))]
end

"""
Return the children, parents, and parents of children (excluding target) as a Set of NodeNames
"""
function markov_blanket(bn::BayesNet, target::NodeName)
	nodeNames = NodeName[]
        for child in children(bn, target)
		append!(nodeNames, parents(bn, child))
                push!(nodeNames, child)
        end
        append!(nodeNames, parents(bn, target))
        return setdiff(Set(nodeNames), Set(NodeName[target]))
end

"""
Whether the BayesNet contains the given edge
"""
function has_edge(bn::BayesNet, parent::NodeName, child::NodeName)::Bool
	u = get(bn.name_to_index, parent, 0)
	v = get(bn.name_to_index, child, 0)
	u != 0 && v != 0 && has_edge(bn.dag, u, v)
end

function enforce_topological_order!(bn::BayesNet)
	dag2, cpds2, name_to_index2 = _enforce_topological_order(bn.dag, bn.cpds, bn.name_to_index)
	bn.dag = dag2
	bn.cpds = cpds2
	bn.name_to_index = name_to_index2
	bn
end
function adding_edge_preserves_acyclicity(parent_list::Vector{Vector{Int}}, u::Int, v::Int)
    n = length(parent_list)
    visited = falses(n)
    propagate = [u]
    while !isempty(propagate)
        x = pop!(propagate)
        if x == v
            return false
        end
        if !visited[x]
            visited[x] = true
            append!(propagate, parent_list[x])
        end
    end
    return true
end

"""
Returns whether the set of node names `x` is d-separated from the set `y` given the set `given`
"""
function is_independent(bn::BayesNet, x::NodeNames, y::NodeNames, given::NodeNames)

	start_node = x[1]
	finish_node = y[1]

	if start_node == finish_node
		return true
	end

	C = Set(given)
	analyzed_nodes = Set()

	# Find all paths from x to y
	paths = []
	path_queue = []
	push!(analyzed_nodes, start_node)
	for next_node in neighbors(bn, start_node)
		if !in(next_node, analyzed_nodes)
			push!(analyzed_nodes, next_node)
			push!(path_queue, [start_node, next_node])
		end
	end
	while (!isempty(path_queue))
		cur_path = pop!(path_queue)
	 	last_node = cur_path[end]
		for next_node in neighbors(bn, last_node)
			if next_node == finish_node
				push!(paths, push!(copy(cur_path), next_node))
			elseif !in(next_node, analyzed_nodes)
				push!(analyzed_nodes, next_node)
				push!(path_queue, push!(copy(cur_path), next_node))
			end
		end
	end

	# Check each path to see if it contains information indicating d-separation
	for path in paths
		is_d_separated = false
		if length(path) == 2
			is_d_separated = true
		else
			# Examine all middle nodes
			for i in 2:(length(path) - 1)
				prev_node = path[i - 1]
				cur_node = path[i]
				next_node = path[i + 1]

				# Check for chain or fork (first or second d-separation criteria)
				if in(cur_node, C)

					# Chain
					if in(cur_node, children(bn, prev_node)) && in(next_node, children(bn, cur_node))
						is_d_separated = true
						break

					# Fork
					elseif in(prev_node, children(bn, cur_node)) && in(next_node, children(bn, cur_node))
						is_d_separated = true
						break
					end

				# Check for v-structure (third d-separation criteria)
				else
					if in(cur_node, children(bn, prev_node)) && in(cur_node, children(bn, next_node))
						descendant_list = descendants(bn, cur_node)
						descendants_in_C = false
						for d in descendant_list
							if in(d, C)
								descendants_in_C = true
								break
							end
						end

						if !descendants_in_C
							is_d_separated = true
							break
						end
					end
				end
			end
		end

		if !is_d_separated
			return false
		end
	end

	# All paths are d-separated, so x and y are conditionally independent.
	return true
end

function Base.push!(bn::BayesNet, cpd::CPD)

	cpdname = name(cpd)
	!haskey(bn.name_to_index, cpdname) || error("A CPD with name $cpdname already exists!")

	add_vertex!(bn.dag)

	push!(bn.cpds, cpd)
	bn.name_to_index[cpdname] = j = length(bn.cpds)

	# add the necessary edges
	for p in parents(cpd)
		i = bn.name_to_index[p]
		add_edge!(bn.dag, i, j)
	end

	!is_cyclic(bn.dag) || error("BayesNet graph is non-acyclic!")
	enforce_topological_order!(bn)
end
function Base.append!(bn::BayesNet, cpds::AbstractVector{C}) where {C<:CPD}
	for cpd in cpds
		push!(bn, cpd)
	end
	bn
end

"""
	delete!(bn::BayesNets, target::NodeName)
Removing cpds will alter the vertex indeces. In particular, removing
the ith cpd will swap i and n and then remove n.
"""
function Base.delete!(bn::BayesNet, target::NodeName)

	if outdegree(bn.dag, bn.name_to_index[target]) > 0
		warn("Deleting a CPD with children!")
	end

	i = bn.name_to_index[target]
	replacement = name(bn.cpds[end])
	bn.cpds[i] = bn.cpds[end]
	pop!(bn.cpds)
	bn.name_to_index[replacement] = i
	delete!(bn.name_to_index, target)
	rem_vertex!(bn.dag, i)
	bn
end

"""
The pdf of a given assignment after conditioning on the values
"""
function CPDs.pdf(bn::BayesNet, assignment::Assignment)
	retval = 1.0
	for cpd in bn.cpds # NOTE: guaranteed in topological order
		retval *= pdf(cpd, assignment)
	end
 	retval
end

"""
The logpdf of a given assignment after conditioning on the values
"""
function CPDs.logpdf(bn::BayesNet, assignment::Assignment)
	retval = 0.0
	for cpd in bn.cpds # NOTE: guaranteed in topological order
		retval += logpdf(cpd, assignment)
	end
 	retval
end



