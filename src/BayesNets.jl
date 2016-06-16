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

	add_edge!,
    has_edge,
# 	remove_edge!,
# 	remove_edges!,

# 	rand_bernoulli_dict,
# 	rand_discrete_dict,

# 	table,
# 	rand_table,
# 	rand_table_weighted,
# 	consistent,,

    table,
    sumout,
    normalize,
    estimate,
    estimate_convergence,

    statistics,
    bayesian_score_component,
    bayesian_score,

    DirichletPrior,
    UniformPrior,
    BDeuPrior

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

include("DiscreteBayesNet/ndgrid.jl")
include("DiscreteBayesNet/factors.jl")
include("DiscreteBayesNet/dirichlet_priors.jl")
include("DiscreteBayesNet/discrete_bayes_net.jl")

# include("factors.jl")
# include("deprecated.jl")

end # module
