const Assignment = Dict{NodeName, Any}

"""
    names(a::Assignment)
Return a vector of NodeNames (aka symbols) for the assignment
"""
Base.names(a::Assignment) = collect(keys(a))

"""
    consistent(a::Assignment, b::Assignment)
True if all shared NodeNames have the same value
"""
function consistent(a::Assignment, b::Assignment)

    for key in keys(a)
        if haskey(b,key) && b[key] != a[key]
            return false
        end
    end

    return true
end