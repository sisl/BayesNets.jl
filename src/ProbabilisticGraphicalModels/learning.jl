"""
    GraphSearchStrategy
An abstract type which defines a graph search strategy for learning probabilistic graphical model structures
These allow: fit(::Type{ProbabilisticGraphicalModel}, data, GraphSearchStrategy)
"""
abstract type GraphSearchStrategy end

"""
    fit(::Type{ProbabilisticGraphicalModel}, data::DataFrame, params::GraphSearchStrategy)
Runs the graph search algorithm to learn a probabilistic graphical model of the provided type from data.
"""
Distributions.fit(pgm::Type{P}, data::DataFrame, params::GraphSearchStrategy) where {P <: ProbabilisticGraphicalModel} = error("fit not defined for $(pgm), DataFrame, and $(typeof(params))")
