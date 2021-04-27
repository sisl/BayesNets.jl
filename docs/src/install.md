# Installation

```julia
Pkg.add("BayesNets");
```
Default visualization of the network structure is provided by the GraphPlot package. However, we recommend using tex-formatted graphs provided by the TikzGraphs package. Installation requirements for TikzGraphs (e.g., PGF/Tikz and pdf2svg) are provided [here](http://nbviewer.ipython.org/github/sisl/TikzGraphs.jl/blob/master/doc/TikzGraphs.ipynb). Simply run using `TikzGraphs` in your Julia session to automatically switch to tex-formatted graphs (thanks to the Requires.jl package).