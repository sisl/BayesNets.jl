VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module BayesNets

using Compat
using Reexport
# using LightXML

include(joinpath("CPDs", "cpds.jl"))
@reexport using BayesNets.CPDs

import LightGraphs: DiGraph, rem_edge!, add_edge!, add_vertex!, has_edge, topological_sort_by_dfs, in_neighbors, out_neighbors, is_cyclic, nv, ne
import TikzGraphs: plot, simple_graph

export
	BayesNet,
	DAG,

    parents,
    children,
    has_edge,
    enforce_topological_order!,

# 	node,
# 	add_node!,
# 	add_nodes!,
	add_edge!,
# 	remove_edge!,
# 	add_edges!,
# 	remove_edges!,
	has_edge
# 	set_CPD!,
# 	set_domain!,
# 	prob,
# 	cpd,
# 	parents,
# 	children,
# 	names,
# 	plot,

# 	rand_bernoulli_dict,
# 	rand_discrete_dict,

# 	table,
# 	rand_table,
# 	rand_table_weighted,
# 	sumout,
# 	normalize,
# 	consistent,
# 	estimate,
# 	estimate_convergence,

# 	prior,
# 	log_bayes_score,
# 	index_data,
# 	statistics,
# 	statistics!,

# 	readxdsl


include("bayes_nets.jl")
include("io.jl")
include("sampling.jl")
include("learning.jl")

# include("ndgrid.jl")
# include("factors.jl")
# include("deprecated.jl")

end # module
