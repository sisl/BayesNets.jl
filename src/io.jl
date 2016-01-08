Base.mimewritable(::MIME"image/svg+xml", b::BayesNet) = true
Base.mimewritable(::MIME"text/html", dfs::Vector{DataFrame}) = true

function plot(b::BayesNet)
	if !isempty(names(b))
		plot(b.dag, AbstractString[string(s) for s in names(b)])
	else
		plot(simple_graph(1), ["Empty Graph"])
	end
end

function Base.writemime(f::IO, a::MIME"image/svg+xml", b::BayesNet)
 	Base.writemime(f, a, plot(b))
end

function Base.writemime(io::IO, a::MIME"text/html", dfs::Vector{DataFrame})
	for df in dfs
		writemime(io, a, df)
	end
end

"""
constructs an array of assignment dictionaries [Dict{Symbol, Assignment}]
assignments are created such that the first node's instantiations change most quickly
"""
function assignment_dicts(
	bn    :: BayesNet,
	nodes :: AbstractVector{Symbol}
	)

	n_nodes = length(nodes)

	@assert(n_nodes > 0)

	domains   = map(pa->domain(bn, pa), nodes)
	nbins     = map(dom->length(dom.elements), domains)
	n_inst    = prod(nbins)
	retval    = Array(Dict{Symbol, Any}, n_inst)
	inst      = ones(Int32, n_nodes)
	inst2assignment = instantiation->Dict([nodes[i]=>domains[i].elements[instantiation[i]] for i in 1:n_nodes])
	retval[1] = inst2assignment(inst)
	for perm  = 2 : n_inst
		# get the next instantiation
		i = 1
		while true
			if inst[i] < nbins[i]
				inst[i] += 1
				inst[1:i-1] = 1
				break
			else
				i += 1
			end
		end
		retval[perm] = inst2assignment(inst)
	end

	retval
end

"""
returns a dict mapping an assignment to the list of probabilities
"""
function discrete_parameter_dict(
	assignments  :: Vector{Dict{Symbol, Any}}, # assumed to be in order
	probs        :: Vector{Float64}, # length ninst_target * length(assignments)
	ninst_target :: Int # number of values in target variable domain
	)

	n_assignments = length(assignments)
	@assert(length(probs) == n_assignments * ninst_target)

	ind = 1
	dict = Dict{Dict{Symbol, Any}, Vector{Float64}}()
	for a in assignments
		dict[a] = probs[ind:ind+ninst_target-1]
		ind += ninst_target
	end

	dict
end

"""
returns a CPD function for a discrete value
"""
function discrete_parameter_function(
	assignments  :: Vector{Dict{Symbol, Any}}, # assumed to be in order
	probs        :: Vector{Float64}, # length ninst_target * length(assignments)
	ninst_target :: Int # number of values in target variable domain
	)

	dict = discrete_parameter_dict(assignments, probs, ninst_target)

	syms = keys(assignments[1])
	return (a)->begin a
		a_extracted = Dict([sym=>a[sym] for sym in syms])
		dict[a_extracted]
	end
end
function readxdsl( filename::AbstractString )
	# Loads a discrete Bayesian Net from XDSL format (SMILE / GeNIe)
	# This currently assumes that the states are Integers
	# Thus, all <state id="???"/> must be integers

	splitext(filename)[2] == ".xdsl" || error("readxdsl only supports .xdsl format")

	xdoc  = parse_file(filename)
	xroot = root(xdoc)
	ces   = get_elements_by_tagname(xroot, "nodes")[1]
	cpts  = collect(child_elements(ces))

	names = Array(Symbol, length(cpts))
	for (i,e) in enumerate(cpts)
		id = attribute(e, "id")
		names[i] = symbol(id)
	end

	BN = BayesNet(names)

	for (i,e) in enumerate(cpts)

		node_sym = names[i]

		# Want to support strings and other types of state IDs, not just integers.
		# for s in get_elements_by_tagname(e, "state")
		# 	attr = convert(ASCIIString, attribute(s, "id"))
		# 	@assert(!isa(match(r"\d", attr), Void), "All state ids must be integers")
		# end

		# set the node's domain
		#states   = [parse(Int, match(r"\d", convert(ASCIIString, attribute(s, "id"))).match) for s in get_elements_by_tagname(e, "state")]
		#states   = [convert(ASCIIString, attribute(s, "id")) for s in get_elements_by_tagname(e, "state")]
		states = get_domain(e)

		n_states = length(states)::Int
		BN.nodes[i].domain = DiscreteDomain(states)

		probs = map(s->parse(Float64, s), split(content(find_element(e, "probabilities"))))

		# set any parents & populate probability table
		parents_elem = get_elements_by_tagname(e, "parents")
		if !isempty(parents_elem)
			parents = map(s->symbol(s), split(content(parents_elem[1])))
			for pa in parents
				add_edge!(BN, pa, node_sym)
			end

			# populate probability table
			reverse!(parents) # because SMILE varies first parent least quickly
			assignments = assignment_dicts(BN, parents)
			parameterFunction = discrete_parameter_function(assignments, probs, n_states)
			set_CPD!(BN, node_sym, CPDs.DiscreteFunction(states, parameterFunction))
		else
			# no parents
			set_CPD!(BN, node_sym, CPDs.DiscreteStatic(states, probs))
		end
	end

	BN
end

function get_domain( e::LightXML.XMLElement )
# Check whether any of the domain elements is a non-integer.  If so, the whole domain will be composed of strings
	
	states   = [parse(Int, match(r"\d", convert(ASCIIString, attribute(s, "id"))).match) for s in get_elements_by_tagname(e, "state")]
	if in(false,[convert(ASCIIString, attribute(s, "id"))==match(r"\d", convert(ASCIIString, attribute(s, "id"))).match for s in get_elements_by_tagname(e, "state")])
		# At least one element of the domain is not an integer:
		states   = [convert(ASCIIString, attribute(s, "id")) for s in get_elements_by_tagname(e, "state")]
	end

	states
end
