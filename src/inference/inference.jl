#
# Main code and big ideas for inference
#
# Also, integration of Factors.jl into BayesNets


abstract AbstractInferenceState

type InferenceState <: AbstractInferenceState
    bn::DiscreteBayesNet
    factor::Factors.Factor
    evidence::Assignment

    """
        InferenceState(bn, query, evidence=Assignment)

    Generates an inference state to be used for inference.
    """
    function InferenceState(bn::DiscreteBayesNet, query::Vector{NodeName},
            evidence::Assignment=Assignment())
        factor = Factor.Factor(query, Float64)

        return new(bn, factor, evidence)
    end

    function InferenceState(bn::DiscreteBayesNet, query::NodeName;
            evidence::Assignment=Assignment())

        return InferenceState(bn, [query], evidence)
    end
end


# THE MOST BASIC ASSUMPTION IS THAT ALL VARIABLES ARE CATEGORICAL AND THEREFORE
# Base.OneTo WORTHY. IF THAT IS VIOLATED, NOTHING WILL WORK
function Factors.Factor(bn::DiscreteBayesNet, name::NodeName,
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

"""
    DataFram(inf)

Return a DataFrame of the probabilities of the query.
"""
function DataFrames.DataFrame(inf::AbstractInferenceState)
    df = DataFrame(inf.factor)
    rename!(df, :v, :p)

    return df
end

