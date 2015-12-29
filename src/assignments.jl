#=
An assignment of the variables in a Bayesian Network is represented as a dictionary.
=#

typealias NodeName Symbol
typealias Assignment Dict

"""
True if all common keys between the two assignments have the same value
TODO: make more efficient
"""
function consistent(a::Assignment, b::Assignment)
    commonKeys = intersect(keys(a), keys(b))
    reduce(&, [a[k] == b[k] for k in commonKeys])
end