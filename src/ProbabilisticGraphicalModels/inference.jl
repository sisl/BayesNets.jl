"""
Abstract type for probability inference
"""
abstract type InferenceMethod end

@inline function _ensure_query_nodes_in_pgm_and_not_in_evidence(query::NodeNames, nodenames::NodeNames, evidence::Assignment)
    isempty(query) && return

    q = first(query)
    (q ∈ nodenames) || throw(ArgumentError("Query $q is not in the probabilistic graphical model"))
    haskey(evidence, q) && throw(ArgumentError("Query $q is part of the evidence"))

    return _ensure_query_nodes_in_pgm_and_not_in_evidence(query[2:end], nodenames, evidence)
end

"""
Type for capturing the inference state
"""
struct InferenceState{PGM<:ProbabilisticGraphicalModel}
    pgm::PGM
    query::NodeNames
    evidence::Assignment

    function InferenceState{PGM}(pgm::PGM, query::NodeNames, evidence::Assignment) where PGM<:ProbabilisticGraphicalModel

        _ensure_query_nodes_in_pgm_and_not_in_evidence(query, names(pgm), evidence)
        return new(pgm, query, evidence)
    end
end
function InferenceState(pgm::PGM, query::NodeNameUnion, evidence::Assignment=Assignment()) where {PGM<:ProbabilisticGraphicalModel}
    query = unique(convert(NodeNames, query))
    return InferenceState{PGM}(pgm, query, evidence)
end


function Base.show(io::IO, inf::InferenceState)
    println(io, "Query: $(inf.query)")
    println(io, "Evidence:")
    for (k, v) in inf.evidence
        println(io, "  $k => $v")
    end
end

"""
    infer(InferenceMethod, InferenceState)
Infer p(query|evidence)
"""
infer(im::InferenceMethod, inf::InferenceState) = error("infer not implemented for $(typeof(im)) and $(typeof(inf))")
infer(im::InferenceMethod, pgm::ProbabilisticGraphicalModel, query::NodeNameUnion; evidence::Assignment=Assignment()) = infer(im, InferenceState(pgm, query, evidence))
