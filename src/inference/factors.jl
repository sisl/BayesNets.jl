#
# Integration of Factors.jl into BayesNets
#
# Get a factor for a BayesNet Node

function Factor(bn::DiscreteBayesNet, name::NodeName,
        evidence::Assignment=Assignment())
    cpd = get(bn, name)
    names = vcat(name, parents(bn, name))
    lengths = ntuple(i -> ncategories(bn, names[i]), length(names))
    dims = map(Factors.CartesianDimension, names, lengths)

    v = Array{Float64}(lengths)
    v[:] = vcat([d.p for d in cpd.distributions]...)
    ft = Factors.Factor(dims, v)

    return ft[evidence]
end

