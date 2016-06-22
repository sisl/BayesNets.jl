# BayesNets

[![Build Status](https://travis-ci.org/sisl/BayesNets.jl.svg?branch=master)](https://travis-ci.org/sisl/BayesNets.jl) [![Coverage Status](https://coveralls.io/repos/sisl/BayesNets.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/sisl/BayesNets.jl?branch=master)

This library supports representation, inference, and learning in Bayesian networks.

Please read the [documentation](http://nbviewer.ipython.org/github/sisl/BayesNets.jl/blob/master/doc/BayesNets.ipynb).

## v1.0.0 Release

We are proud to announce a new major release for BayesNets.jl.
The new approach integrates with Distributions.jl, always ensures BayesNets types are fully defined, and moves the discrete-network-specific material to parameterized DiscreteBayesNets.
We now have support for structure learning, easy parameter learning, etc.

Unfortunately the new API required breaking backwards compatability.
The new API should be easier to understand and use.
Please consult the [documentation](http://nbviewer.ipython.org/github/sisl/BayesNets.jl/blob/master/doc/BayesNets.ipynb) for examples.
