module BayesNets

import LightGraphs: DiGraph, Edge, rem_edge!, add_edge!, has_edge, topological_sort_by_dfs, in_edges, src, dst, in_neighbors, is_cyclic
import TikzGraphs: plot
import Base: rand, select
import DataFrames: DataFrame, groupby, array, isna

export BayesNet, addEdge!, removeEdge!, addEdges!, removeEdges!, CPD, CPDs, prob, setCPD!, pdf, rand, randBernoulliDict, randDiscreteDict, table, domain, Assignment, *, sumout, normalize, select, randTable, NodeName, consistent, estimate, randTableWeighted, estimateConvergence, isValid, hasEdge, probvec
export Domain, BinaryDomain, DiscreteDomain, RealDomain, domain, cpd, parents, setDomain!, plot

include("assignments.jl")
include("utils.jl")
include("domains.jl")
include("cpds.jl"); using BayesNets.CPDs
include("bayesnets.jl")
include("sampling.jl")
include("ndgrid.jl")
include("learning.jl")
include("io.jl")

end # module
