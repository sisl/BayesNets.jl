#=
A Bayesian Network (BN) represents a probability distribution
over a set of variables, P(x₁, x₂, ..., xₙ)
It leverages relations between variables in order to efficiently encode it.
A BN is defined by a directed acyclic graph in which each node is a variable
and contains an associated probability distribution P(xⱼ | parents(xⱼ))
=#

typealias DAG DiGraph

DAG(n) = DiGraph(n)

# TODO: if CPDs give you domains, why do we need to store separate domains here?
type BayesNetNode
	name::NodeName # the symbol corresponding to the node
	domain::Domain # the domain for this variable
	cpd::CPD # the conditional probability distribution P(x|parents(x))
end

type BayesNet
	dag::DAG # the directed acyclic graph, represented as a LightGraphs.DiGraph
	nodes::Vector{BayesNetNode} # the nodes for the BayesNet
	name_to_index::Dict{NodeName,Int} # NodeName → index in dag and nodes

	function BayesNet()
		retval = new()
		retval.dag = DAG(0)
		retval.nodes = BayesNetNode[]
		retval.name_to_index = Dict{NodeName, Int}()
		retval
	end
	function BayesNet(dag::DAG, nodes::AbstractVector{BayesNetNode})

		@assert(nv(dag) == length(nodes))

		retval = new()
		retval.dag = dag
		retval.nodes = nodes

		retval.name_to_index = Dict{NodeName, Int}()
		for (i, node) in enumerate(nodes)
			retval.name_to_index[node.name] = i
		end

		retval
	end
	function BayesNet(
		nodes::AbstractVector{BayesNetNode},
		edges::AbstractVector{Tuple{NodeName, NodeName}},
		)

		retval = new()
		retval.dag = DAG(length(nodes))
		retval.nodes = nodes
		retval.name_to_index = Dict{NodeName, Int}()
		for (i, node) in enumerate(nodes)
			retval.name_to_index[node.name] = i
		end

		for (parent, child) in edges
			u = retval.name_to_index[parent]
		    v = retval.name_to_index[child]
			add_edge!(retval.dag, u, v)
		end

		retval
	end

	"""
	Generate a BayesNet with the given names
	The DAG will be edgeless, and each variable
	will be given a Bernoulli and binary domain
	"""
	function BayesNet(names::AbstractVector{NodeName})
		n = length(names)

		retval = new()
		retval.dag = DiGraph(n)
		retval.nodes = Array(BayesNetNode, n)
		for (i,name) in enumerate(names)
			retval.nodes[i] = BayesNetNode(name, BINARY_DOMAIN, CPDs.Bernoulli())
		end

		retval.name_to_index = Dict{NodeName, Int}()
		for (i, node) in enumerate(retval.nodes)
			retval.name_to_index[node.name] = i
		end

		retval
	end
end

node(bn::BayesNet, name::NodeName) = bn.nodes[bn.name_to_index[name]]
CPDs.domain(bn::BayesNet, name::NodeName) = node(bn, name).domain
cpd(bn::BayesNet, name::NodeName) = node(bn, name).cpd
function names(bn::BayesNet)
	retval = Array(NodeName, length(bn.nodes))
	for (i,node) in enumerate(bn.nodes)
		retval[i] = node.name
	end
	retval
end

Base.isvalid(bn::BayesNet) = !is_cyclic(bn.dag)

"""
Returns the parents as a list of NodeNames
"""
function parents(bn::BayesNet, name::NodeName)
	i = bn.name_to_index[name]
	NodeName[bn.nodes[j].name for j in in_neighbors(bn.dag, i)]
end

"""
Returns the children as a list of NodeNames
"""
function children(bn::BayesNet, name::NodeName)
	i = bn.name_to_index[name]
	NodeName[bn.nodes[j].name for j in out_neighbors(bn.dag, i)]
end

function has_edge(bn::BayesNet, parent::NodeName, child::NodeName)
    u = bn.name_to_index[parent]
    v = bn.name_to_index[child]
	has_edge(bn.dag, u, v)
end

function add_edge!(bn::BayesNet, parent::NodeName, child::NodeName)
	u = bn.name_to_index[parent]
    v = bn.name_to_index[child]
	add_edge!(bn.dag, u, v)
	bn
end
function add_edges!(bn::BayesNet, pairs::AbstractVector{Tuple{NodeName, NodeName}})
	for p in pairs
  		add_edge!(bn, p[1], p[2])
	end
	bn
end

function add_node!(bn::BayesNet, node::BayesNetNode)
	add_vertex!(bn.dag)
	push!(bn.nodes, node)
	bn.name_to_index[node.name] = length(bn.nodes)
	bn
end
function add_nodes!(bn::BayesNet, nodes::AbstractVector{BayesNetNode})
	for node in nodes
		add_node!(bn, node)
	end
	bn
end

function remove_edge!(bn::BayesNet, parent::NodeName, child::NodeName)
	#=
	NOTE:
	it would be nice to use a more efficient implementation
	see discussion here: https://github.com/JuliaLang/Graphs.jl/issues/73
	and here: https://github.com/JuliaLang/Graphs.jl/pull/87
	=#

	u = bn.name_to_index[parent]
    v = bn.name_to_index[child]
	rem_edge!(bn.dag, u, v)
	bn
end
function remove_edges!(bn::BayesNet, pairs::AbstractVector{Tuple{NodeName, NodeName}})
	for p in pairs
		remove_edge!(bn, p[1], p[2])
	end
	bn
end

function set_domain!(bn::BayesNet, name::NodeName, dom::Domain)
	mynode = node(bn, name)
	mynode.domain = dom
	bn
end

function set_CPD!(bn::BayesNet, name::NodeName, cpd::CPD)
	mynode = node(bn, name)
	mynode.cpd = cpd
	bn
end

"""
Computes the probability of a given assignment
NOTE: if all variables are discrete, this is a discrete probability
	  if all variables are continuous, this is a probability density

      prob(BN) = ∏ P(xⱼ | parents(xⱼ))
"""
function prob(bn::BayesNet, assignment::Assignment)
	retval = 1.0
	for node in bn.nodes
		pdf_func = pdf(node.cpd, assignment)
		retval *= pdf_func(assignment[node.name])
	end
 	retval
end

"""
TODO: rename table() to factor()?
Constructs the CPD factor associated with the given node in the BayesNet
"""
function table(bn::BayesNet, name::NodeName)

	d = DataFrame()
    c = cpd(bn, name)
    names = push!(parents(bn, name), name)

    nparents = length(names)-1
    if nparents > 0
        A = ndgrid([domain(bn, name).elements for name in names]...)
        for (i,name2) in enumerate(names)
            d[name2] = vec(A[i])
        end
    else
        d[name] = domain(bn, name).elements
    end

    p = ones(size(d,1)) # the probability column
    for i in 1:size(d,1)
        ownValue = d[i,length(names)]
        assignment = Dict([names[j]=>d[i,j] for j in 1:nparents])
        pdf_func = pdf(c, assignment)
        p[i] = pdf_func(ownValue)
    end
    d[:p] = p
    d
end

table(bn::BayesNet, name::NodeName, a::Assignment) = select(table(bn, name), a)
