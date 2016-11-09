# TODO use factors.jl

"""
TODO description
"""
function variable_elimination(bn::DiscreteBayesNet, elimination_order::Vector{Symbol}, query::Vector{Symbol}; evidence::Assignment=Assignment())
    """
    TODO algorithm
    TODO unit tests under test/
    
    Algorithm Steps:
    Make sure parameters are valid
    Take all cpds and convert the to factors as defined in factors.jl
    Use the select function in factors.jl to remove every symbol in the evidence argument from each factor
    For each variable in the elimination_order:
        Find all factors containing that variable
        Multiply them together with the * function defined in factors.jl
        Use the sumout function in factors.jl to marginalize out the variable
    Multiply the remaining factors together using the * function defined in factors.jl
    Return the result of the multiplication
    """
end
