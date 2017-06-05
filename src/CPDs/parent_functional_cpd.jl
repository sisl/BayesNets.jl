type ParentFunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    accessor::Function # calling this gives you the distribution from the assignment
end
ParentFunctionalCPD{D}(target::NodeName, accessor::Function) = new{D}(target, NodeName[], accessor)

name(cpd::ParentFunctionalCPD) = cpd.target
parents(cpd::ParentFunctionalCPD) = cpd.parents
@define_call ParentFunctionalCPD
@compat (cpd::ParentFunctionalCPD)(a::Assignment) = cpd.accessor(a, parents(cpd))
