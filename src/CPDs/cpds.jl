#=
A CPD is a Conditional Probability Distribution
In general, theyrepresent distribtions of the form P(X|Y)
Each node in a Bayesian Network is associated with a variable,
and contains the CPD relating that var to its parents, P(x | parents(x))
=#

# module CPDs

using Reexport
@reexport using Distributions
@reexport using DataFrames

# export
#     CPD,                         # the abstract CPD type

#     Assignment,                  # variable assignment type, complete or partial, for a Bayesian Network
#     NodeName,                    # variable name type

#     name,                        # obtain the name of the CPD
#     distribution,                # returns the CPD's distribution type
#     trained,                     # whether the CPD has been trained
#     pdf,                         # probability density function or probability distribution function (continuous or discrete)
#     learn!                       # train a CPD based on data

typealias NodeName Symbol
typealias Assignment Dict

abstract CPD{D<:Distribution}

#=
Each CPD must implement:

    name(cpd::CPD)                                      # name of the variable
    parents(cpd::CPD)                                   # list of parents (Symbols)
    distribution(cpd::CPD)                              # get the internal distribution for the CPD

    condition!(cpd::CPD, assignment::Assignment)        # update the conditional distribution with the observation
    Distributions.fit(CPD, data, target [, parents=[]])
=#

disttype{D}(cpd::CPD{D}) = D
Base.rand(cpd::CPD, a::Assignment) = rand(pdf(cpd, a))

Distributions.pdf(cpd::CPD, a::Assignment) = pdf(distribution(cpd), a[name(cpd)])
function pdf!(cpd::CPD, a::Assignment)
    condition!(cpd, a)
    pdf(cpd, a)
end

###########################

type CPDCore{D<:Distribution}
    name::NodeName
    parents::Vector{NodeName}
    d::D
end

###########################

type StaticCPD{D<:Distribution} <: CPD{D}
    core::CPDCore{D}
end

name(cpd::StaticCPD) = cpd.core.name
parents(cpd::StaticCPD) = cpd.core.parents
distribution(cpd::StaticCPD) = cpd.core.d

condition!(cpd::StaticCPD, a::Assignment) = cpd.core.d # do nothing
function Distributions.fit{D}(::Type{StaticCPD{D}}, data::DataFrame, target::NodeName, parents::Vector{NodeName}=NodeName[])
    d = fit(D, data[target])
    core = CPDCore(target, parents, d)
    StaticCPD(core)
end

###########################

include("utils.jl")
# include("categorical_cpd.jl")
# include("linear_gaussian.jl")

# end # module CPDs