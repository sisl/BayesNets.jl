#=
A Bayesian Network (BN) represents a probability distribution
over a set of variables, P(x₁, x₂, ..., xₙ)
It leverages relations between variables in order to efficiently encode the joint distribution.
A BN is defined by a directed acyclic graph in which each node is a variable
and contains an associated conditional probability distribution P(xⱼ | parents(xⱼ))
=#

typealias DAG DiGraph

function _build_dag_from_cpds{T<:CPD}(
	cpds::AbstractVector{T},
	name_to_index::Dict{NodeName, Int}
	)

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
function _enforce_topological_order{T<:CPD}(
	dag::DAG,
	cpds::AbstractVector{T},
	name_to_index::Dict{NodeName, Int},
	)

	topo = topological_sort_by_dfs(dag)

	cpds2 = Array(T, length(cpds))
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

type BayesNet{T<:CPD}
	dag::DAG # nodes are in topological order
	cpds::Vector{T} # the CPDs associated with each node in the dag
	name_to_index::Dict{NodeName,Int} # NodeName → index in dag and cpds
end
BayesNet() = BayesNet(DAG(0), CPD[], Dict{NodeName, Int}())
BayesNet{T <: CPD}(::Type{T}) = BayesNet(DAG(0), T[], Dict{NodeName, Int}())
function BayesNet{T <: CPD}(cpds::AbstractVector{T})

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
	retval = Array(NodeName, length(bn))
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
	NodeName[name(bn.cpds[j]) for j in out_neighbors(bn.dag, i)]
end

function has_edge(bn::BayesNet, parent::NodeName, child::NodeName)
    u = bn.name_to_index[parent]
    v = bn.name_to_index[child]
	has_edge(bn.dag, u, v)
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
function Base.append!{C<:CPD}(bn::BayesNet, cpds::AbstractVector{C})
	for cpd in cpds
		push!(bn, cpd)
	end
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
CPDs.pdf(bn::BayesNet, pair::Pair{NodeName}...) = pdf(bn, Assignment(pair))

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
CPDs.logpdf(bn::BayesNet, pair::Pair{NodeName}...) = logpdf(bn, Assignment(pair))

"""
The logpdf of a set of assignment after conditioning on the values
"""
function CPDs.logpdf(bn::BayesNet, df::DataFrame)

	logl = 0.0

	a = Assignment()
	varnames = names(bn)
	for i in 1 : nrow(df)

		for name in varnames
			a[name] = df[i, name]
		end

		logl += logpdf(bn, a)
	end

	logl
end

"""
The pdf of a set of assignment after conditioning on the values
"""
CPDs.pdf(bn::BayesNet, df::DataFrame) = exp(logpdf(bn, df))

