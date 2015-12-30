
typealias DAG DiGraph

DAG(n) = DiGraph(n)

type BayesNet
	dag::DAG
	cpds::Vector{CPD}
	index::Dict{NodeName,Int}
	names::Vector{NodeName}
	domains::Vector{Domain}

	function BayesNet(names::Vector{NodeName})
	    n = length(names)
	    index = [names[i]=>i for i in 1:n]

	    cpds = Array(CPD, n)
	    for i in 1:n
	    	cpds[i] = BernoulliCPD()
	    end


	    # cpds = CPD[CPDs.BernoulliCPD() for i in 1:n]
	    domains = Domain[BinaryDomain() for i in 1:n] # default to binary domain
	    new(DiGraph(length(names)), cpds, index, names, domains)
	end
end

CPDs.domain(b::BayesNet, name::NodeName) = b.domains[b.index[name]]
cpd(b::BayesNet, name::NodeName) = b.cpds[b.index[name]]

function parents(b::BayesNet, name::NodeName)
	i = b.index[name]
	NodeName[b.names[j] for j in in_neighbors(b.dag, i)]
end

isValid(b::BayesNet) = !is_cyclic(b.dag)

function hasEdge(bn::BayesNet, sourceNode::NodeName, destNode::NodeName)
    u = bn.index[sourceNode]
    v = bn.index[destNode]
	return has_edge(bn.dag, u, v)
end

function addEdge!(bn::BayesNet, sourceNode::NodeName, destNode::NodeName)
	i = bn.index[sourceNode]
	j = bn.index[destNode]
	add_edge!(bn.dag, i, j)
	bn
end

function removeEdge!(bn::BayesNet, sourceNode::NodeName, destNode::NodeName)
	#=
	it would be nice to use a more efficient implementation
	see discussion here: https://github.com/JuliaLang/Graphs.jl/issues/73
	and here: https://github.com/JuliaLang/Graphs.jl/pull/87
	=#

	i = bn.index[sourceNode]
	j = bn.index[destNode]
	rem_edge!(bn.dag, i, j)
	bn
end

function addEdges!(bn::BayesNet, pairs)
	for p in pairs
  		addEdge!(bn, p[1], p[2])
	end
	bn
end

function removeEdges!(bn::BayesNet, pairs)
	for p in pairs
		removeEdge!(bn, p[1], p[2])
	end
	bn
end

function setDomain!(bn::BayesNet, name::NodeName, dom::Domain)
	i = bn.index[name]
	bn.domains[i] = dom
end

function setCPD!(bn::BayesNet, name::NodeName, cpd::CPD)
	i = bn.index[name]
	bn.cpds[i] = cpd
	bn.domains[i] = domain(cpd)
	nothing
end

function prob(bn::BayesNet, assignment::Assignment)
 	prod([pdf(bn.cpds[i], assignment)(assignment[bn.names[i]]) for i = 1:length(bn.names)])
end

function table(bn::BayesNet, name::NodeName)
    edges = in_edges(bn.dag, bn.index[name])
    names = [bn.names[src(e)] for e in edges]
    push!(names, name)
    c = cpd(bn, name)
    d = DataFrame()
    if length(edges) > 0
        A = ndgrid([domain(bn, name).elements for name in names]...)
        i = 1
        for name in names
            d[name] = A[i][:]
            i = i + 1
        end
    else
        d[name] = domain(bn, name).elements
    end
    p = ones(size(d,1))
    for i = 1:size(d,1)
        ownValue = d[i,length(names)]
        a = [names[j]=>d[i,j] for j = 1:(length(names)-1)]
        p[i] = pdf(c, a)(ownValue)
    end
    d[:p] = p
    d
end

table(bn::BayesNet, name::NodeName, a::Assignment) = select(table(bn, name), a)