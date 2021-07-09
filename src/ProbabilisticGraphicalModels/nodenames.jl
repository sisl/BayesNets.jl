const NodeName = Symbol


const NodeNames = AbstractVector{NodeName}
const NodeNameUnion = Union{NodeName, NodeNames}

nodeconvert(::Type{NodeNames}, names::NodeNameUnion) = names


nodeconvert(::Type{NodeNames}, name::NodeName) = [name]
