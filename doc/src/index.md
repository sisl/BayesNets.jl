# [BayesNets](https://github.com/sisl/BayesNets.jl)

*A Julia package for Bayesian Networks*

A Bayesian Network (BN) represents a probability distribution over a set of variables, ![](https://latex.codecogs.com/gif.download?P%28x_1%2C%20x_2%2C%20%5Cldots%2C%20x_n%29). Bayesian networks leverage variable relations in order to efficiently decompose the joint distribution into smaller conditional probability distributions.

A BN is defined by a directed acyclic graph and a set of conditional probability distributions. Each node in the graph corresponds to a variable ![](https://latex.codecogs.com/gif.download?x_i) and is associated with a conditional probability distribution ![](https://latex.codecogs.com/gif.download?P%28x_i%20%5Cmid%20%5Ctext%7Bparents%7D%28x_i%29%29).

### Table of Contents

```@contents
Pages = ["install.md", "usage.md"]
```