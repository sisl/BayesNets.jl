"""
    bayesian_score(g::DAG, names::Vector{Symbol}, data::DataFrame, prior::DirichletPrior=UniformPrior())

Compute the bayesian score for graph structure `g`, with the data in `data`. `names` containes a symbol corresponding to each vertex in `g` that is the name of a column in `data`.

Note that every entry in data must be an integer greater than 0
"""
function bayesian_score(g::DAG, names::Vector{Symbol}, data::DataFrame, prior::DirichletPrior=UniformPrior())
    n = nv(g)
    bincounts = Array(Int, n)
    datamat = Array(Int, ncol(data), nrow(data))
    for i in 1:n
        datamat[i,:] = data[names[i]]
        bincounts[i] = infer_number_of_instantiations(convert(Vector{Int}, data[i]))
    end

    return bayesian_score(g.badjlist, bincounts, datamat, prior)
end
