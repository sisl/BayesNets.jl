Base.mimewritable(::MIME"image/svg+xml", bn::BayesNet) = true
Base.mimewritable(::MIME"text/html", dfs::Vector{DataFrame}) = true

function plot(bn::BayesNet)
	if !isempty(names(bn))
		plot(bn.dag, AbstractString[string(s) for s in names(bn)]) # NOTE: sometimes the same var shows up twice
	else
		plot(DiGraph(1), ["Empty Graph"])
	end
end

@compat function Base.show(f::IO, a::MIME"image/svg+xml", bn::BayesNet)
 	show(f, a, plot(bn))
end

@compat function Base.show(io::IO, a::MIME"text/html", dfs::Vector{DataFrame})
	for df in dfs
		writemime(io, a, df)
	end
end

import LightXML

"""
    readxdsl( filename::AbstractString )
Return a DiscreteBayesNet read from the xdsl file
"""
function readxdsl( filename::AbstractString )

	# This currently assumes that the states are Integers
	# Thus, all <state id="???"/> must be integers

	splitext(filename)[2] == ".xdsl" || error("readxdsl only supports .xdsl format")

	xdoc  = LightXML.parse_file(filename)
	xroot = LightXML.root(xdoc)
	ces   = LightXML.get_elements_by_tagname(xroot, "nodes")[1]
	cpts  = collect(LightXML.child_elements(ces))

	varnames = Array(Symbol, length(cpts))
	for (i,e) in enumerate(cpts)
		id = LightXML.attribute(e, "id")
		varnames[i] = Symbol(id)
	end

	bn = DiscreteBayesNet()

	for (i,e) in enumerate(cpts)

		node_sym = varnames[i]

		for s in LightXML.get_elements_by_tagname(e, "state")
			attr = convert(String, LightXML.attribute(s, "id"))
			@assert(!isa(match(r"\d", attr), Void), "All state ids must be integers")
		end

		# set the node's domain
		states   = [parse(Int, match(r"\d", convert(String, LightXML.attribute(s, "id"))).match) for s in LightXML.get_elements_by_tagname(e, "state")]

		probs = Float64[parse(Float64, s) for s in split(LightXML.content(LightXML.find_element(e, "probabilities")))]

		# set any parents & populate probability table
		parents_elem = LightXML.get_elements_by_tagname(e, "parents")
		if !isempty(parents_elem)
			parents = NodeName[Symbol(s) for s in split(LightXML.content(parents_elem[1]))]

			# populate probability table
			reverse!(parents) # because SMILE varies first parent least quickly
            parental_ncategories = _get_parental_ncategories(bn, parents)
            k = length(states)
            Q = prod(parental_ncategories)
            distributions = Array(Categorical, Q)
            for q in 1:Q
                hi = k*q
                lo = hi - k + 1
                distributions[q] = Categorical{Float64}(probs[lo:hi])
            end

            push!(bn, DiscreteCPD(node_sym, parents, parental_ncategories, distributions))
		else
			# no parents
            push!(bn, DiscreteCPD(node_sym, probs))
		end
	end

	bn
end
