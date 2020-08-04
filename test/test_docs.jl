using NBInclude
import Pkg; Pkg.add("TikzGraphs")

let
    @nbinclude(joinpath(dirname(@__DIR__), "doc", "BayesNets.ipynb"))
end
