"""
A CPD for which the distribution never changes.
    target: name of the CPD's variable
    parents: list of parent variables.
    d: a Distributions.jl distribution

While a StaticCPD can have parents, their assignments will not affect the distribution.
"""
type StaticCPD{D} <: CPD{D}
    target::NodeName
    parents::Vector{NodeName}
    d::D
end
StaticCPD(target::NodeName, d::Distribution) = StaticCPD(target, NodeName[], d)

name(cpd::StaticCPD) = cpd.target
parents(cpd::StaticCPD) = cpd.parents

Base.call(cpd::StaticCPD, a::Assignment) = cpd.d # no update
function Distributions.fit{D<:Distribution}(::Type{StaticCPD{D}},
    data::DataFrame,
    target::NodeName,
    parents::Vector{NodeName}=NodeName[],
    )

    d = fit(D, data[target])
    StaticCPD(target, parents, d)
end
