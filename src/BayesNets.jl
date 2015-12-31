module BayesNets

using LightXML

import LightGraphs: DiGraph, Edge, rem_edge!, add_edge!, add_vertex!, has_edge, topological_sort_by_dfs, in_edges, src, dst, in_neighbors, is_cyclic, nv, ne
import TikzGraphs: plot
import DataFrames: DataFrame, groupby, array, isna, names

export
	BayesNet,
	BayesNetNode,
	DAG,

	node,
	add_node!,
	add_nodes!,
	add_edge!,
	remove_edge!,
	add_edges!,
	remove_edges!,
	has_edge,
	set_CPD!,
	set_domain!,
	prob,
	cpd,
	pdf,
	parents,
	names,
	plot,

	rand_bernoulli_dict,
	rand_discrete_dict,

	table,
	rand_table,
	rand_table_weighted,
	sumout,
	normalize,
	consistent,
	estimate,
	estimate_convergence,

	prior,
	log_bayes_score,
	index_data,
	statistics,
	statistics!,

	readxdsl

using Reexport
include("cpds.jl");
@reexport using BayesNets.CPDs

include("ndgrid.jl")
include("assignments.jl")
include("factors.jl")
include("bayes_nets.jl")
include("sampling.jl")
include("learning.jl")
include("io.jl")

end # module
