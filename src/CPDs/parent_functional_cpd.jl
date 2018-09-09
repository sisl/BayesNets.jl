mutable struct ParentFunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    accessor::Function # calling this gives you the distribution from the assignment

    function ParentFunctionalCPD{D}(target::NodeName, accessor::Function) where D
        new(target, NodeName[], accessor)
    end
    function ParentFunctionalCPD{D}(target::NodeName, parents::NodeNames, accessor::Function) where D
        new(target, parents, accessor)
    end
end

name(cpd::ParentFunctionalCPD) = cpd.target
parents(cpd::ParentFunctionalCPD) = cpd.parents
(cpd::ParentFunctionalCPD)(a::Assignment) = cpd.accessor(a, parents(cpd))
(cpd::ParentFunctionalCPD)() = (cpd)(Assignment()) # cpd()
(cpd::ParentFunctionalCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)
