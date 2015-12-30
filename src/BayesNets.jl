module BayesNets

import LightGraphs: DiGraph, Edge, rem_edge!, add_edge!, has_edge, topological_sort_by_dfs, in_edges, src, dst, in_neighbors, is_cyclic
import TikzGraphs: plot
import DataFrames: DataFrame, groupby, array, isna, names

export
	BayesNet,
	BayesNetNode,

	node,
	add_edge!,
	remove_edge!,
	add_edges!,
	remove_edges!,
	has_edge,
	set_CPD!,
	set_domain!,
	prob,
	pdf,
	parents,
	names,
	plot,

	randBernoulliDict,
	randDiscreteDict,

	table,
	randTable,
	randTableWeighted,
	sumout,
	normalize,
	consistent,
	estimate,
	estimateConvergence,
	log_bayes_score

using Reexport
include("cpds.jl");
@reexport using BayesNets.CPDs

include("assignments.jl")
include("factors.jl")
include("bayesnets.jl")
include("sampling.jl")
include("ndgrid.jl")
include("learning.jl")
include("io.jl")

end # module
