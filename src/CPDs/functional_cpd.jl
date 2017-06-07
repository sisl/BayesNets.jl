type FunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    accessor::Function # calling this gives you the distribution from the assignment
end
FunctionalCPD{D}(target::NodeName, accessor::Function) = new{D}(target, NodeName[], accessor)

name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
@define_call FunctionalCPD
@compat (cpd::FunctionalCPD)(a::Assignment) = cpd.accessor(a)