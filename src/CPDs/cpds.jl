#=
A CPD is a Conditional Probability Distribution
In general, they represent distribtions of the form P(X|Y)
Each node in a Bayesian Network is associated with a variable,
and contains the CPD relating that var to its parents, P(x | parents(x))
=#

module CPDs

using Discretizers
using Distributions
using DataFrames
using Reexport
using Printf
using Random
using Statistics

include(joinpath("../ProbabilisticGraphicalModels", "ProbabilisticGraphicalModels.jl"))
using BayesNets.CPDs.ProbabilisticGraphicalModels

using IterTools

export
    CPD,                           # the abstract CPD type

    StaticCPD,                     # static distribution (never uses parental information)
    FunctionalCPD,                 # for implementing quick and easy custom CPDs
    ParentFunctionalCPD,           # for implementing quick and easy custom CPDs that only use the parents
    CategoricalCPD,                # a table lookup based on discrete parental assignment
    LinearGaussianCPD,             # Normal with linear mean
    ConditionalLinearGaussianCPD,  # a LinearGaussianCPD lookup based on discrete parental assignment
    DiscreteCPD,                   # a typealias to CategoricalCPD{Categorical}

    NamedCategorical,              # a custom distribution, a Categorical with named values

    name,                          # obtain the name of the CPD
    parents,                       # obtain the parents in the CPD
    parentless,                    # whether the given variable is parentless
    disttype,                      # returns the CPD's distribution type
    nparams,                       # returns the number of free parameters required for the distribution

    # utils
    strip_arg,
    required_func,
    sub2ind_vec,
    infer_number_of_instantiations

#############################################

include("utils.jl")

#############################################

abstract type CPD{D<:Distribution} end

"""
    name(cpd::CPD)
Return the NodeName for the variable this CPD is defined for.
"""
@required_func name(cpd::CPD)

"""
    parents(cpd::CPD)
Return the parents for this CPD as a vector of NodeName.
"""
@required_func parents(cpd::CPD)

"""
    fit(::Type{CPD}, data::DataFrame, target::NodeName, parents::NodeNames)
Construct a CPD for target by fitting it to the provided data
"""
@required_func Distributions.fit(cpdtype::Type{CPD}, data::DataFrame, target::NodeName, parents::NodeNames)
@required_func Distributions.fit(cpdtype::Type{CPD}, data::DataFrame, target::NodeName)

"""
    nparams(cpd::CPD)
Return the number of free parameters that needed to be estimated for the CPD
"""
@required_func nparams(cpd::CPD)

"""
    parentless(cpd::CPD)
Return whether this CPD has parents.
"""
parentless(cpd::CPD) = isempty(parents(cpd))

"""
    disttype(cpd::CPD)
Return the type of the CPD's distribution
"""
disttype(cpd::CPD{D}) where {D} = D

"""
    rand(cpd::CPD)
Condition and then draw from the distribution
"""
Base.rand(cpd::CPD, a::Assignment) = rand(cpd(a))
Base.rand(cpd::CPD, pair::Pair{NodeName}...) = rand(cpd, Assignment(pair))

"""
    pdf(cpd::CPD)
Condition and then return the pdf
"""
Distributions.pdf(cpd::CPD, a::Assignment) = pdf(cpd(a), a[name(cpd)])
Distributions.pdf(cpd::CPD, pair::Pair{NodeName}...) = pdf(cpd, Assignment(pair))

"""
    logpdf(cpd::CPD)
Condition and then return the logpdf
"""
Distributions.logpdf(cpd::CPD, a::Assignment) = logpdf(cpd(a), a[name(cpd)])
Distributions.logpdf(cpd::CPD, pair::Pair{NodeName}...) = logpdf(cpd, Assignment(pair))

"""
    logpdf(cpd::CPD, data::DataFrame)
Return the logpdf across the dataset
"""
function Distributions.logpdf(cpd::CPD, data::DataFrame)
    retval = 0.0
    a = Assignment()
    for i in 1 : nrow(data)
        get!(a, cpd, data, i)
        retval += logpdf(cpd, a)
    end
    retval
end

"""
    pdf(cpd::CPD, data::DataFrame)
Return the pdf across the dataset
"""
Distributions.pdf(cpd::CPD, data::DataFrame) = exp(logpdf(cpd, data))

###########################

"""
    get!(a::Assignment, b::Assignment)
Modify and return the assignment to contain the ith entry
"""
function Base.get!(a::Assignment, cpd::CPD, data::DataFrame, i::Int)
    target = name(cpd)
    a[target] = data[i,target]
    for j in parents(cpd)
        a[j] = data[i,j]
    end
    a
end

###########################

include("named_categorical.jl")

include("static_cpd.jl")
include("functional_cpd.jl")
include("parent_functional_cpd.jl")
include("categorical_cpd.jl")
include("linear_gaussian_cpd.jl")
include("conditional_linear_gaussian_cpd.jl")

end # module CPDs
