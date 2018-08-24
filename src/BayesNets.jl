module BayesNets

using Reexport
using Parameters
using Random
using LinearAlgebra
using Printf
using IterTools
using Dates
using SpecialFunctions
using SparseArrays

include(joinpath("CPDs", "cpds.jl"))
@reexport using BayesNets.CPDs
@reexport using BayesNets.CPDs.ProbabilisticGraphicalModels

import Base: *, /, +, -
import DataStructures: PriorityQueue, peek
import BayesNets.CPDs.ProbabilisticGraphicalModels: markov_blanket, is_independent, infer
import StatsBase: sample, Weights
import TikzGraphs: plot
import LightGraphs: DiGraph, add_edge!, rem_edge!,
       add_vertex!, rem_vertex!, has_edge,
       edges, topological_sort_by_dfs, inneighbors,
       outneighbors, neighbors, is_cyclic, nv, ne,
       outdegree, bfs_tree, dst

export
    BayesNet,
    DAG,

    parents,
    children,
    neighbors,
    descendants,
    has_edge,
    enforce_topological_order!,

    add_edge!,
    has_edge,

    rand_cpd,
    BayesNetSampler,
    gibbs_sample,
    GibbsSampler,

    # tables, formerly known as factors; the DataFrames based approach
    # TODO remove
    Table,
    table,
    sumout,
    normalize,
    # estimate_convergence,
    readxdsl,

    # generate BNs
    rand_discrete_bn,
    rand_bn_inference,
    get_sprinkler_bn,
    get_sat_fail_bn,
    get_asia_bn,

    # Factors
    Factor,
    pattern,
    reducedim!,

    # Sampling
    BayesNetSampler,
    DirectSampler,
    RejectionSampler,
    LikelihoodWeightedSampler,

    get_weighted_sample,
    get_weighted_sample!,

    # Inference
    InferenceState,
    InferenceMethod,
    ExactInference,
    LikelihoodWeightingInference,
    LoopyBelief,
    GibbsSamplingNodewise,
    GibbsSamplingFull,

    infer,

    DirichletPrior,
    UniformPrior,
    BDeuPrior,

    ScoringFunction,
    ScoreComponentCache,
    NegativeBayesianInformationCriterion,
    score_component,
    score_components,

    GraphSearchStrategy,
    K2GraphSearch,
    GreedyHillClimbing,
    GreedyThickThinning,
    ScanGreedyHillClimbing,

    statistics,
    index_data,
    adding_edge_preserves_acyclicity,
    bayesian_score_component,
    bayesian_score_components,
    bayesian_score


include("bayes_nets.jl")
include("io.jl")
include("sampling.jl")
include("gibbs.jl")
include("learning.jl")
include("gen_bayes_nets.jl")

include(joinpath("DiscreteBayesNet", "ndgrid.jl"))
include(joinpath("DiscreteBayesNet", "tables.jl"))
include(joinpath("DiscreteBayesNet", "dirichlet_priors.jl"))
include(joinpath("DiscreteBayesNet", "discrete_bayes_net.jl"))
include(joinpath("DiscreteBayesNet", "structure_scoring.jl"))
include(joinpath("DiscreteBayesNet", "greedy_hill_climbing.jl"))
include(joinpath("DiscreteBayesNet", "scan_greedy_hill_climbing.jl"))
include(joinpath("DiscreteBayesNet", "io.jl"))

include(joinpath("Factors", "factors.jl"))
include(joinpath("Inference", "inference.jl"))

end # module
