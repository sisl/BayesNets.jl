"""
Abstract type for probability inference
"""
abstract InferenceMethod

"""
Infer p(query|evidence)
 - inference on a DiscreteBayesNet will always return a DataFrame factor over the evidence variables
"""
infer(im::InferenceMethod, bn::BayesNet, query::Vector{NodeName}; evidence::Assignment=Assignment()) = error("infer not implemented for $(typeof(im))")
infer(im::InferenceMethod, bn::BayesNet, query::NodeName; evidence::Assignment=Assignment()) = infer(im, bn, [query]; evidence=evidence)

