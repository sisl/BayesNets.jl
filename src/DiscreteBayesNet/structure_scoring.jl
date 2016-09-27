"""
    bayesian_score(g::DAG, names::Vector{Symbol}, data::DataFrame[, ncategories::Vector{Int}[, prior::DirichletPrior]])

Compute the bayesian score for graph structure `g`, with the data in `data`. `names` containes a symbol corresponding to each vertex in `g` that is the name of a column in `data`. `ncategories` is a vector of the number of values that each variable in the Bayesian network can take.

Note that every entry in data must be an integer greater than 0
"""
function bayesian_score(g::DAG,
                        names::Vector{Symbol},
                        data::DataFrame,
                        ncategories::Vector{Int}=Int[infer_number_of_instantiations(convert(Vector{Int}, data[n])) for n in names],
                        prior::DirichletPrior=UniformPrior())
    datamat = Array(Int, ncol(data), nrow(data))
    for i in 1:nv(g)
        datamat[i,:] = data[names[i]]
    end

    return bayesian_score(badj(g), ncategories, datamat, prior)
end
