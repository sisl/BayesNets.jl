#
# Main code and big ideas for inference
#

@inline function _ensure_query_nodes_in_bn_and_not_in_evidence(qs::NodeNames, nodes::NodeNames, ev::Assignment)
    isempty(qs) && return

    q = first(qs)
    (q in nodes) || throw(ArgumentError("Query $q is not in the bayes net"))
    haskey(ev, q) && throw(ArgumentError("Query $q is part of the evidence"))

    return _ensure_query_nodes_in_bn_and_not_in_evidence(qs[2:end], nodes, ev)
end

immutable InferenceState
    bn::DiscreteBayesNet
    query::NodeNames
    evidence::Assignment

    """
        InferenceState(bn, query, evidence=Assignment)

    Generates an inference state to be used for inference.
    """
    function InferenceState(bn::DiscreteBayesNet, query::NodeNameUnion, evidence::Assignment=Assignment())
        query = unique(convert(NodeNames, query))
        _ensure_query_nodes_in_bn_and_not_in_evidence(query, names(bn), evidence)

        return new(bn, query, evidence)
    end
end

Base.names(inf::InferenceState) = names(inf.bn)

function Base.show(io::IO, inf::InferenceState)
    println(io, "Query: $(inf.query)")
    println(io, "Evidence:")
    for (k, v) in inf.evidence
        println(io, "  $k => $v")
    end
end

###############################

"""
Abstract type for probability inference
"""
abstract InferenceMethod

"""
Infer p(query|evidence)
 - inference on a DiscreteBayesNet will always return a DataFrame factor over the evidence variables
"""
infer(im::InferenceMethod, inf::InferenceState) = error("infer not implemented for $(typeof(im)) and $(typeof(inf))")
infer(im::InferenceMethod, bn::BayesNet, query::NodeNameUnion; evidence::Assignment=Assignment()) = infer(im, InferenceState(bn, query, evidence))


