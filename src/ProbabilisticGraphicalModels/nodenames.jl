typealias NodeName Symbol
typealias NodeNames AbstractVector{NodeName}
typealias NodeNameUnion Union{NodeName, NodeNames}
    
Base.convert(::Type{NodeNames}, name::NodeName) = [name]