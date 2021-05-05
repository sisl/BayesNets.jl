# Concepts


## Bayesian Networks

A Bayesian Network (BN) represents a probability distribution over a set of variables, ``P(x_1, x_2, \ldots, x_n)``. 
Bayesian networks leverage variable relations in order to efficiently decompose the joint distribution into smaller conditional probability distributions.

A BN is defined by a directed acyclic graph and a set of conditional probability distributions. Each node in the graph corresponds to a variable ``x_i`` and is associated with a conditional probability distribution ``P(x_i \mid \text{parents}(x_i))``.

