"""
A CPD for which the distribution never changes.
    target: name of the CPD's variable
    parents: list of parent variables.
    d: a Distributions.jl distribution

While a StaticCPD can have parents, their assignments will not affect the distribution.
"""
mutable struct StaticCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    d::D
end
StaticCPD(target::NodeName, d::Distribution) = StaticCPD(target, NodeName[], d)

name(cpd::StaticCPD) = cpd.target
parents(cpd::StaticCPD) = cpd.parents
(cpd::StaticCPD)(a::Assignment) = cpd.d # no update
(cpd::StaticCPD)() = (cpd)(Assignment()) # cpd()
(cpd::StaticCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function Distributions.fit(::Type{StaticCPD{D}},
                           data::DataFrame,
                           target::NodeName,
                           parents::NodeNames=NodeName[],
                           ) where {D<:Distribution}

    d = fit(D, data[target])
    StaticCPD(target, parents, d)
end

nparams(cpd::StaticCPD) = paramcount(params(cpd.d))
