function Distributions.fit(::Type{BayesNet}, data::DataFrame, dag::DAG, cpd_types::Vector{DataType})

    length(cpd_types) == nv(dag) || throw(DimensionMismatch("dag and cpd_types must have the same length"))

    cpds = Array(CPD, length(cpd_types))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        C = cpd_types[i]
        parents = tablenames[in_neighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
function Distributions.fit{C<:CPD}(::Type{BayesNet}, data::DataFrame, dag::DAG, ::Type{C})

    cpds = Array(C, nv(dag))
    tablenames = names(data)
    for (i, target) in enumerate(tablenames)
        parents = tablenames[in_neighbors(dag, i)]
        cpds[i] = fit(C, data, target, parents)
    end

    BayesNet(cpds)
end
Distributions.fit{T<:CPD}(::Type{BayesNet{T}}, data::DataFrame, dag::DAG) = fit(BayesNet, data, dag, T)