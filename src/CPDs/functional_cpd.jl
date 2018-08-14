mutable struct FunctionalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    accessor::Function # calling this gives you the distribution from the assignment

    function FunctionalCPD{D}(target::NodeName, accessor::Function) where D
        new(target, NodeName[], accessor)
    end
    function FunctionalCPD{D}(target::NodeName, parents::NodeNames, accessor::Function) where D
        new(target, parents, accessor)
    end
end


name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
(cpd::FunctionalCPD)(a::Assignment) = cpd.accessor(a)
(cpd::FunctionalCPD)() = (cpd)(Assignment()) # cpd()
(cpd::FunctionalCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)
