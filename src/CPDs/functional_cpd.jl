type FunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::Vector{NodeName}
    accessor::Function # calling this gives you the distribution from the assignment

    FunctionalCPD(target::NodeName, accessor::Function) = new(target, NodeName[], accessor)
    FunctionalCPD(target::NodeName, parents::Vector{NodeName}, accessor::Function) = new(target, parents, accessor)
end

name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
@define_call FunctionalCPD
@compat (cpd::FunctionalCPD)(a::Assignment) = cpd.accessor(a)