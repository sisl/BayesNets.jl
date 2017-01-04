#
# Main code and big ideas for inference
#

# An implementation of an AbstractInferenceState has:
#  bn::DiscreteBayesNet
#  query::Vector{NodeName}
#  evidence::Assignment
# and a constructor that accepts just those three arguments in that order
abstract AbstractInferenceState

immutable InferenceState <: AbstractInferenceState
    bn::DiscreteBayesNet
    query::Vector{NodeName}
    evidence::Assignment

    """
        InferenceState(bn, query, evidence=Assignment)

    Generates an inference state to be used for inference.
    """
    function InferenceState(bn::DiscreteBayesNet, query::NodeNames,
            evidence::Assignment=Assignment())
        if isa(query, NodeName)
            query = [query]
        end

        # check if any queries aren't in the network
        inds = indexin(query, names(bn))
        zero_loc = findnext(inds, 0, 1)
        if zero_loc != 0
            throw(ArgumentError("$(query[zero_loc]) is not in the bayes net"))
        end

        return new(bn, query, evidence)
    end
end

Base.names(inf::AbstractInferenceState) = names(inf.bn)

function Base.show(io::IO, inf::AbstractInferenceState)
    println(io, "Query: $(inf.query)")
    println(io, "Evidence:")
    for (k, v) in inf.evidence
        println(io, "  $k => $v")
    end
end

