# Usage

##Representation

Bayesian Networks are represented with the `BayesNet` type. This type contains the directed acyclic graph (a LightTables.DiGraph) and a list of conditional probability distributions (a list of CPDs).
Here we construct the BayesNet $a \\rightarrow b$, with Gaussians $a$ and $b$:

a = \\mathcal{N}(0,1) \\qquad b = \\mathcal{N}(2a +3,1)\n

```
bn = BayesNet()
push!(bn, StaticCPD(:a, Normal(1.0)))
push!(bn, LinearGaussianCPD(:b, [:a], [2.0], 3.0, 1.0))
```



##Likelihood

A Bayesian Network represents a joint probability distribution, $P(x_1, x_2, \\ldots, x_n)$.
Assignments are represented as dictionaries mapping variable names (Symbols) to variable values.
We can evaluate probabilities as we would with Distributions.jl, only we use exclamation points as we modify the internal state when we condition:

```pdf(bn, :a=>0.5, :b=>2.0) # evaluate the probability density```

We can also evaluate the likelihood of a dataset:

```
data = DataFrame(a=[0.5,1.0,2.0], b=[4.0,5.0,7.0])
pdf(bn, data)    #  0.00215
logpdf(bn, data) # -6.1386;
```

Or the likelihood for a particular cpd:

```
pdf(cpdB, data)    #  0.006
logpdf(cpdB, data) # -5.201
```

##Sampling

Assignments can be sampled from a `BayesNet`.

```rand(bn)```
```
Dict{Symbol,Any} with 2 entries:
  :a => 1.20808
  :b => 4.93954
```

In general, sampling can be done according to `rand(BayesNet, BayesNetSampler, nsamples)` to produce a table of samples, `rand(BayesNet, BayesNetSampler)` to produce a single Assignment, or `rand!(Assignment, BayesNet, BayesNetSampler)` to modify an assignment in-place.
New samplers need only implement `rand!`.
The functions above default to the `DirectSampler`, which samples the variables in topographical order.

Rejection sampling can be used to draw samples that are consistent with a provided assignment:
```
bn = BayesNet()
push!(bn, StaticCPD(:a, Categorical([0.3,0.7])))
push!(bn, StaticCPD(:b, Categorical([0.6,0.4])))
push!(bn, CategoricalCPD{Bernoulli}(:c, [:a, :b], [2,2], [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(1.0), Bernoulli(0.4)]))
```

```
rand(bn, RejectionSampler(:c=>1), 5)
```
# there is a table here 



##Parameter Learning
BayesNets.jl supports parameter learning for an entire graph.

```# specify each node's CPD type individually
fit(BayesNet, data, (:a=>:b), [StaticCPD{Normal}, LinearGaussianCPD])
```
```# specify a single CPD type for all nodes
fit(BayesNet, data, (:a=>:b), LinearGaussianCPD)
```
Fitting can be done for specific BayesNets types as well:

```data = DataFrame(c=[1,1,1,1,2,2,2,2,3,3,3,3],
b=[1,1,1,2,2,2,2,1,1,2,1,1],
a=[1,1,1,2,1,1,2,1,1,2,1,1])

fit(DiscreteBayesNet, data, (:a=>:b, :a=>:c, :b=>:c))
```

Fitting a ```DiscreteCPD```, which is a ```CategoricalCPD{Categorical}```, can be done with a specified number of categories. This prevents cases where your test data does not provide an example for every category.

```cpd = fit(DiscreteCPD, DataFrame(a=[1,2,1,2,2]), :a, ncategories=3);
cpd = fit(DiscreteCPD, data, :b, [:a], parental_ncategories=[3], target_ncategories=3);
```

## Inference

Inference methods for discrete Bayesian networks can be used via the `infer` method:

```julia
bn = DiscreteBayesNet()
push!(bn, DiscreteCPD(:a, [0.3,0.7]))
push!(bn, DiscreteCPD(:b, [0.2,0.8]))
push!(bn, DiscreteCPD(:c, [:a, :b], [2,2], 
        [Categorical([0.1,0.9]),
         Categorical([0.2,0.8]),
         Categorical([1.0,0.0]),
         Categorical([0.4,0.6]),
        ]))

ϕ = infer(bn, :c, evidence=Assignment(:b=>1))
```

Several inference methods are available. Exact inference is the default.

| `Inference Method` | Description |
| ------------------ | ----------- |
| `ExactInference`   | Performs exact inference using discrete factors and variable elimination|
| `LikelihoodWeightingInference` | Approximates p(query \ evidence) with N weighted samples using likelihood weighted sampling |
| `LoopyBelief` | The loopy belief propagation algorithm |
| `GibbsSamplingNodewise` | Gibbs sampling where each iteration changes one node |
| `GibbsSamplingFull` | Gibbs sampling where each iteration changes all nodes |
