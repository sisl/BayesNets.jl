Pkg.add("RDatasets")
Pkg.add("NBInclude")
using NBInclude

let
    nbinclude(Pkg.dir("BayesNets", "doc", "BayesNets.ipynb"))
end
