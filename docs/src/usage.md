# Usage

```julia
using Random
Random.seed!(0) # seed the random number generator to 0, for a reproducible demonstration
using BayesNets
using TikzGraphs # required to plot tex-formatted graphs (recommended), otherwise GraphPlot.jl is used
```

## Representation

Bayesian Networks are represented with the `BayesNet` type. This type contains the directed acyclic graph (a LightTables.DiGraph) and a list of conditional probability distributions (a list of CPDs).
Here we construct the BayesNet $a \rightarrow b$, with Gaussians $a$ and $b$:

```math
a = \mathcal{N}(0,1) \qquad b = \mathcal{N}(2a +3,1)
```

```julia
bn = BayesNet()
push!(bn, StaticCPD(:a, Normal(1.0)))
push!(bn, LinearGaussianCPD(:b, [:a], [2.0], 3.0, 1.0))
```

## Conditional Probability Distributions

Conditional Probablity Distributions, $P(x_i \mid \text{parents}(x_i))$, are defined in BayesNets.CPDs. Each CPD knows its own name, the names of its parents, and is associated with a distribution from Distributions.jl.

| `CPDForm`                      | Description |
| ------------------------------ | ----------- |
| `StaticCPD`                    | Any `Distributions.distribution`; indepedent of any parents |
| `FunctionalCPD`                | Allows for a CPD defined with a custom eval function |
| `ParentFunctionalCPD`          | Modification to `FunctionalCPD` allowing the parent values to be passed in |
| `CategoricalCPD`               | Categorical distribution, assumes integer parents in $1:N$ |
| `LinearGaussianCPD`            | Linear Gaussian, assumes target and parents are numeric |
| `ConditionalLinearGaussianCPD` | A linear Gaussian for each discrete parent instantiation|

Each CPD can be learned from data using `fit`.
Here we learn the same network as above.

```julia
a = randn(100)
b = randn(100) .+ 2*a .+ 3

data = DataFrame(a=a, b=b)
cpdA = fit(StaticCPD{Normal}, data, :a)
cpdB = fit(LinearGaussianCPD, data, :b, [:a])

bn2 = BayesNet([cpdA, cpdB])
```
Each `CPD` implements four functions:

* `name(cpd)` - obtain the name of the variable target variable
* `parents(cpd)` - obtain the list of parents
* `nparams(cpd` - obtain the number of free parameters in the CPD
* `cpd(assignment)` - allows calling `cpd()` to obtain the conditional distribution
* `Distributions.fit(Type{CPD}, data, target, parents)`

Several functions conveniently condition and then produce their return values:

```julia
rand(cpdB, :a=>0.5) # condition and then sample
pdf(cpdB, :a=>1.0, :b=>3.0) # condition and then compute pdf(distribution, 3)
logpdf(cpdB, :a=>1.0, :b=>3.0) # condition and then compute logpdf(distribution, 3);
```

The NamedCategorical distribution allows for String or Symbol return values. The FunctionalCPD allows for crafting quick and simple CPDs:

```julia
bn2 = BayesNet()
push!(bn2, StaticCPD(:sighted, NamedCategorical([:bird, :plane, :superman], [0.40, 0.55, 0.05])))
push!(bn2, FunctionalCPD{Bernoulli}(:happy, [:sighted], a->Bernoulli(a == :superman ? 0.95 : 0.2)))
```

Variables can be removed by name using `delete!`. A warning will be issued when removing a CPD with children.

```julia
delete!(bn2, :happy)
```

## Likelihood

A Bayesian Network represents a joint probability distribution, $P(x_1, x_2, \ldots, x_n)$.
Assignments are represented as dictionaries mapping variable names (Symbols) to variable values.
We can evaluate probabilities as we would with Distributions.jl, only we use exclamation points as we modify the internal state when we condition:

```pdf(bn, :a=>0.5, :b=>2.0) # evaluate the probability density```

We can also evaluate the likelihood of a dataset:

```julia
data = DataFrame(a=[0.5,1.0,2.0], b=[4.0,5.0,7.0])
pdf(bn, data)    #  0.00215
logpdf(bn, data) # -6.1386;
```

Or the likelihood for a particular cpd:

```julia
pdf(cpdB, data)    #  0.006
logpdf(cpdB, data) # -5.201
```

## Sampling

Assignments can be sampled from a `BayesNet`.

```julia
rand(bn)
```

In general, sampling can be done according to `rand(BayesNet, BayesNetSampler, nsamples)` to produce a table of samples, `rand(BayesNet, BayesNetSampler)` to produce a single Assignment, or `rand!(Assignment, BayesNet, BayesNetSampler)` to modify an assignment in-place.
New samplers need only implement `rand!`.
The functions above default to the `DirectSampler`, which samples the variables in topographical order.

Rejection sampling can be used to draw samples that are consistent with a provided assignment:
```julia
bn = BayesNet()
push!(bn, StaticCPD(:a, Categorical([0.3,0.7])))
push!(bn, StaticCPD(:b, Categorical([0.6,0.4])))
push!(bn, CategoricalCPD{Bernoulli}(:c, [:a, :b], [2,2], [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(1.0), Bernoulli(0.4)]))
```

```julia
rand(bn, RejectionSampler(:c=>1), 5)
```
### there is a table here 



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

#TODO ADD IMAGE RESULT
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

```julia
ϕ = infer(GibbsSamplingNodewise(), bn, [:a, :b], evidence=Assignment(:c=>2))
```
## Structure Learning

We demonstrate structure learning through an example using an iris dataset.

```julia
using Discretizers
using RDatasets
iris = dataset("datasets", "iris")
names(iris)
data = DataFrame(
    SepalLength = iris[!,:SepalLength],
    SepalWidth = iris[!,:SepalWidth],
    PetalLength = iris[!,:PetalLength],
    PetalWidth = iris[!,:PetalWidth],
    Species = encode(CategoricalDiscretizer(iris[!,:Species]), iris[!,:Species]),
)
```

Here we use the K2 structure learning algorithm which runs in polynomial time but requires that we specify a topological node ordering.

```julia
parameters = K2GraphSearch([:Species, :SepalLength, :SepalWidth, :PetalLength, :PetalWidth], 
                       ConditionalLinearGaussianCPD,
                       max_n_parents=2)
fit(BayesNet, data, parameters)
```

CPD types can also be specified per-node. Note that complete CPD definitions are required - simply using `StaticCPD` is insufficient as you need the target distribution type as well, as in `StaticCPD{Categorical}`.

Changing the ordering will change the structure.

```julia
CLG = ConditionalLinearGaussianCPD
parameters = K2GraphSearch([:Species, :PetalLength, :PetalWidth, :SepalLength, :SepalWidth], 
                        [StaticCPD{Categorical}, CLG, CLG, CLG, CLG],
                        max_n_parents=2)
fit(BayesNet, data, parameters)
```

A `ScoringFunction` allows for extracting a scoring metric for a CPD given data. The negative BIC score is implemented in `NegativeBayesianInformationCriterion`.

A `GraphSearchStrategy` defines a structure learning algorithm. The K2 algorithm is defined through `K2GraphSearch` and `GreedyHillClimbing` is implemented for discrete Bayesian networks and the Bayesian score:

```julia
data = DataFrame(c=[1,1,1,1,2,2,2,2,3,3,3,3], 
                 b=[1,1,1,2,2,2,2,1,1,2,1,1],
                 a=[1,1,1,2,1,1,2,1,1,2,1,1])
parameters = GreedyHillClimbing(ScoreComponentCache(data), max_n_parents=3, prior=UniformPrior())
bn = fit(DiscreteBayesNet, data, parameters)
```

We can specify the number of categories for each variable in case it cannot be correctly inferred:

```julia
bn = fit(DiscreteBayesNet, data, parameters, ncategories=[3,3,2])
```

A whole suite of features are supported for DiscreteBayesNets. Here, we illustrate the following:

1. Obtain a list of counts for a node
2. Obtain sufficient statistics from a discrete dataset
3. Obtain the factor table for a node
4. Obtain a factor table matching a particular assignment

We also detail obtaining a bayesian score for a network structure in the next section.

```julia
count(bn, :a, data) # 1
statistics(bn.dag, data) # 2
table(bn, :b) # 3
table(bn, :c, :a=>1) # 4
```

## Bayesian Score for a Network Structure

The bayesian score for a discrete-valued BayesNet can can be calculated based only on the structure and data (the CPDs do not need to be defined beforehand). This is implemented with a method of ``bayesian_score`` that takes in a directed graph, the names of the nodes and data.

```julia
data = DataFrame(c=[1,1,1,1,2,2,2,2,3,3,3,3], 
                 b=[1,1,1,2,2,2,2,1,1,2,1,1],
                 a=[1,1,1,2,1,1,2,1,1,2,1,1])
g = DAG(3)
add_edge!(g,1,2); add_edge!(g,2,3); add_edge!(g,1,3)
bayesian_score(g, [:a,:b,:c], data)
```

## Reading from XDSL

Discrete Bayesian Networks can be read from the .XDSL file format.

```julia
bn = readxdsl(joinpath(dirname(pathof(BayesNets)), "..", "test", "sample_bn.xdsl"))
```
