const NodeName = Symbol
const NodeNames = AbstractVector{NodeName}
const NodeNameUnion = Union{NodeName, NodeNames}

Base.convert(::Type{NodeNames}, name::NodeName) = [name]