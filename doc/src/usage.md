# Usage

## Parameter Learning
BayesNets.jl supports parameter learning for an entire graph.

```# specify each node's CPD type individually
fit(BayesNet, data, (:a=>:b), [StaticCPD{Normal}, LinearGaussianCPD])
```
#TODO ADD IMAGE RESULT
```# specify a single CPD type for all nodes
fit(BayesNet, data, (:a=>:b), LinearGaussianCPD)
```
#TODO ADD IMAGE RESULT
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

CPD types can also be specified per-node. Note that complete CPD definitions are required - simply using ``StaticCPD`` is insufficient as you need the target distribution type as well, as in ``StaticCPD{Categorical}``.

Changing the ordering will change the structure.

```julia
CLG = ConditionalLinearGaussianCPD
parameters = K2GraphSearch([:Species, :PetalLength, :PetalWidth, :SepalLength, :SepalWidth], 
                        [StaticCPD{Categorical}, CLG, CLG, CLG, CLG],
                        max_n_parents=2)
fit(BayesNet, data, parameters)
```

A ``ScoringFunction`` allows for extracting a scoring metric for a CPD given data. The negative BIC score is implemented in ``NegativeBayesianInformationCriterion``.

A ``GraphSearchStrategy`` defines a structure learning algorithm. The K2 algorithm is defined through ``K2GraphSearch`` and ``GreedyHillClimbing`` is implemented for discrete Bayesian networks and the Bayesian score:

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

## Reading from XDSL

Discrete Bayesian Networks can be read from the .XDSL file format.

```julia
bn = readxdsl(joinpath(dirname(pathof(BayesNets)), "..", "test", "sample_bn.xdsl"))
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
