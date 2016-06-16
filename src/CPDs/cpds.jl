#=
A CPD is a Conditional Probability Distribution
In general, they represent distribtions of the form P(X|Y)
Each node in a Bayesian Network is associated with a variable,
and contains the CPD relating that var to its parents, P(x | parents(x))
=#

module CPDs

using Reexport
@reexport using Distributions
@reexport using DataFrames

export
    CPD,                         # the abstract CPD type
    CPDForm,                     # describes how CPDs are updated and learned

    Assignment,                  # variable assignment type, complete or partial, for a Bayesian Network
    NodeName,                    # variable name type

    StaticCPD,                   # static distribution (never uses parental information)
    CategoricalCPD,
    LinearGaussianCPD,

    pdf!,                        # condition and obtain the pdf
    logpdf!,                     # condition and obtain the logpdf
    name,                        # obtain the name of the CPD
    parents,                     # obtain the parents in the CPD
    parentless,                  # whether the given variable is parentless
    distribution,                # returns the CPD's distribution type
    condition!,                  # update the conditional distribution with the observation

    sub2ind_vec,
    infer_number_of_instantiations,
    consistent

typealias NodeName Symbol
typealias Assignment Dict{Symbol, Any}

#############################################

abstract CPDForm
type CPD{D<:Distribution, C<:CPDForm}
    name::NodeName
    parents::Vector{NodeName}
    d::D
    form::C
end
function CPD{D<:Distribution, C<:CPDForm}(
    name::NodeName,
    d::D,
    form::C,
    )

    CPD{D,C}(name, NodeName[], d, form)
end

name(cpd::CPD) = cpd.name
parents(cpd::CPD) = cpd.parents
parentless(cpd::CPD) = isempty(cpd.parents)
distribution(cpd::CPD) = cpd.d

Base.rand(cpd::CPD) = rand(distribution(cpd))
Base.rand!(cpd::CPD, a::Assignment) = rand(condition!(cpd, a))

Distributions.pdf(cpd::CPD, a::Assignment) = pdf(cpd.d, a[cpd.name])
Distributions.logpdf(cpd::CPD, a::Assignment) = logpdf(cpd.d, a[cpd.name])
function pdf!(cpd::CPD, a::Assignment)
    condition!(cpd, a)
    pdf(cpd.d, a[cpd.name])
end
function logpdf!(cpd::CPD, a::Assignment)
    condition!(cpd, a)
    logpdf(cpd.d, a[cpd.name])
end

#=
Each CPDForm must implement:
    condition!{D, C}(cpd::CPD{D,C}, assignment)         - update the CPD distribution based on the assignment
    fit{D, C}(::Type{CPD{D,C}}, data, target, parents)  - fit the CPDForm (and possible CPD.d) based on data
=#


###########################

type StaticCPD <: CPDForm end

condition!{D,C<:StaticCPD}(cpd::CPD{D,C}, a::Assignment) = cpd.d # no update
function Distributions.fit{D,C<:StaticCPD}(::Type{CPD{D,C}},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName}=NodeName[],
    )

    d = fit(D, data[target])
    CPD(target, parents, d, StaticCPD())
end

###########################

include("utils.jl")
include("categorical_cpd.jl")
include("linear_gaussian_cpd.jl")

end # module CPDs
