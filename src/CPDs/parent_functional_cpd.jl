type ParentFunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    accessor::Function # calling this gives you the distribution from the assignment

    ParentFunctionalCPD(target::NodeName, accessor::Function) = new(target, NodeName[], accessor)
    ParentFunctionalCPD(target::NodeName, parents::NodeNames, accessor::Function) = new(target, parents, accessor)
end

name(cpd::ParentFunctionalCPD) = cpd.target
parents(cpd::ParentFunctionalCPD) = cpd.parents
@define_call ParentFunctionalCPD
@compat (cpd::ParentFunctionalCPD)(a::Assignment) = cpd.accessor(a, parents(cpd))
