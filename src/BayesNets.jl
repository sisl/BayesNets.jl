VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module BayesNets

using Compat
using Reexport

include(joinpath("CPDs", "cpds.jl"))
@reexport using BayesNets.CPDs

import Base.Collections: PriorityQueue, peek
import Iterators: subsets, product
import TikzGraphs: plot
import Factors
import LightGraphs: DiGraph, add_edge!, rem_edge!,
       add_vertex!, rem_vertex!, has_edge,
       edges, topological_sort_by_dfs, in_neighbors,
       out_neighbors, neighbors, is_cyclic, nv, ne,
       outdegree, badj, bfs_tree

export
    BayesNet,
    DAG,

    parents,
    children,
    neighbors,
    descendants,
    markov_blanket,
    has_edge,
    enforce_topological_order!,

    add_edge!,
    has_edge,

    rand_cpd,
    BayesNetSampler,
    gibbs_sample,
    GibbsSampler,

    table,
    sumout,
    normalize,
    estimate_convergence,
    readxdsl,

    # Sampler interface
    BayesNetSampler,
    DirectSampler,
    RejectionSampler,
    WeightedSampler,

    # generate BNs
    rand_discrete_bn,
    bn_inference_init,
    get_sprinkler_bn,
    get_sat_fail_bn,
    get_asia_bn,

    # inference
    AbstractInferenceState,
    InferenceState,
    GibbsInferenceState,
    exact_inference,
    likelihood_weighting,
    gibbs_sampling,
    gibbs_sampling_full_iter,
    loopy_belief,
# TODO swap out the above for the below###################
    exact_inference_inf,
    likelihood_weighting_inf,

    DirichletPrior,
    UniformPrior,
    BDeuPrior,

    ScoringFunction,
    ScoreComponentCache,
    NegativeBayesianInformationCriterion,
    score_component,
    score_components,

    # structure learning
    GraphSearchStrategy,
    K2GraphSearch,
    GreedyHillClimbing,
    GreedyThickThinning,
    ScanGreedyHillClimbing,

    statistics,
    index_data,
    adding_edge_preserves_acyclicity,
    is_independent,
    bayesian_score_component,
    bayesian_score_components,
    bayesian_score


include("bayes_nets.jl")
include("io.jl")
include("sampling.jl")
include("learning.jl")
include("gibbs.jl")
include("gen_bayes_nets.jl")

include(joinpath("DiscreteBayesNet", "ndgrid.jl"))
include(joinpath("DiscreteBayesNet", "factors.jl")) # TODO SWAP OUT #####################
include(joinpath("DiscreteBayesNet", "dirichlet_priors.jl"))
include(joinpath("DiscreteBayesNet", "discrete_bayes_net.jl"))
include(joinpath("DiscreteBayesNet", "structure_scoring.jl"))
include(joinpath("DiscreteBayesNet", "greedy_hill_climbing.jl"))
include(joinpath("DiscreteBayesNet", "scan_greedy_hill_climbing.jl"))

include(joinpath("inference", "inference.jl"))
include(joinpath("inference", "exact.jl"))
include(joinpath("inference", "gibbs.jl"))
include(joinpath("inference", "likelihood.jl"))
include(joinpath("inference", "loopy_belief.jl"))

end # module

