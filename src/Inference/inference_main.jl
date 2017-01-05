#
# Main code and big ideas for inference
#

# An implementation of an AbstractInferenceState has:
#  bn::DiscreteBayesNet
#  query::Vector{NodeName}
#  evidence::Assignment
# and a constructor that accepts just those three arguments in that order
abstract AbstractInferenceState

# make sure query nodes are in the bn and not in the evidence
@inline function _ckq(qs::Vector{NodeName}, nodes::Vector{NodeName},
        ev::Assignment)
    isempty(qs) && return

    q = first(qs)
    (q in nodes) || throw(ArgumentError("Query $q is not in the bayes net"))
    haskey(ev, q) && throw(ArgumentError("Query $q is part of the evidence"))

    return _ckq(qs[2:end], nodes, ev)
end

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
        query = _sandims(query)
        _ckq(query, names(bn), evidence)

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

