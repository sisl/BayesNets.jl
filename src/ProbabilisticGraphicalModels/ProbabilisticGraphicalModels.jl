# __precompile__()

"""
Provides a basic interface for defining and working with probabilistic graphical models
"""
module ProbabilisticGraphicalModels

using Random
using Reexport
@reexport using Distributions
@reexport using DataFrames

export
    ProbabilisticGraphicalModel,

    Assignment,                    # variable assignment type, complete or partial, for a Bayesian Network
    NodeName,                      # variable name type
    NodeNames,                     # vector of names
    NodeNameUnion,                 # either a NodeName or NodeNames

    Sampler,
    GraphSearchStrategy,

    # Inference
    InferenceMethod,
    InferenceState,
    infer,

    # Common Functions that are not in Base or required packages
    markov_blanket,
    is_independent,
    consistent

include("nodenames.jl")
include("assignments.jl")

abstract type ProbabilisticGraphicalModel end

Base.get(pgm::ProbabilisticGraphicalModel, i::Int) = error("get(pgm, Int) not implemented for $(typeof(pgm))!")
Base.get(pgm::ProbabilisticGraphicalModel, name::NodeName) = error("get(pgm, NodeName) not implemented for $(typeof(pgm))!")
Base.get(pgm::ProbabilisticGraphicalModel, names::NodeNames) = [get(pgm, name) for name in names]
Base.get(pgm::ProbabilisticGraphicalModel, names::Base.AbstractSet{NodeName}) = get(pgm, collect(names))

"""
    length(PGM)
Returns the number of variables in the probabilistic graphical model
"""
Base.length(pgm::ProbabilisticGraphicalModel) = error("length not implemented for $(typeof(pgm))!")

"""
    names(PGM)
Returns a list of NodeNames
"""
Base.names(pgm::ProbabilisticGraphicalModel) = error("names not implemented for $(typeof(pgm))!")

"""
markov_blanket(PGM)
Returns the list of NodeNames forming the Markov blanket for the PGM
"""
markov_blanket(pgm::ProbabilisticGraphicalModel) = error("markov_blanket not implemented for $(typeof(pgm))!")

"""
is_independent(PGM, x::NodeNames, y::NodeNames, given::NodeNames)
Returns whether the set of node names `x` is d-separated from the set `y` given the set `given`
"""
is_independent(pgm::ProbabilisticGraphicalModel, x::NodeNames, y::NodeNames, given::NodeNames) = error("is_independent not implemented for $(typeof(pgm))!")

"""
The pdf of a given assignment after conditioning on the values
"""
Distributions.pdf(pgm::ProbabilisticGraphicalModel, assignment::Assignment) = error("pdf not implemented for $(typeof(pgm))!")
Distributions.pdf(pgm::ProbabilisticGraphicalModel, pair::Pair{NodeName}...) = pdf(pgm, Assignment(pair))

"""
The logpdf of a given assignment after conditioning on the values
"""
Distributions.logpdf(pgm::ProbabilisticGraphicalModel, assignment::Assignment) = error("logpdf not implemented for $(typeof(pgm))!")
Distributions.logpdf(pgm::ProbabilisticGraphicalModel, pair::Pair{NodeName}...) = logpdf(pgm, Assignment(pair))

"""
The logpdf of a set of assignment after conditioning on the values
"""
function Distributions.logpdf(pgm::ProbabilisticGraphicalModel, df::DataFrame)

    logl = 0.0

    a = Assignment()
    nodenames = names(pgm)
    for i in 1 : nrow(df)

        for name in nodenames
            a[name] = df[i, name]
        end

        logl += logpdf(pgm, a)
    end

    return logl
end

"""
The pdf of a set of assignments after conditioning on the values
"""
Distributions.pdf(pgm::ProbabilisticGraphicalModel, df::DataFrame) = exp(logpdf(pgm, df))


include("sampling.jl")
include("learning.jl")
include("inference.jl")


end # module ProbabilisticGraphicalModels

