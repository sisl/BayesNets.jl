push!(LOAD_PATH, "../src")
import Pkg
Pkg.develop(path=".")
Pkg.add("TikzPictures")
Pkg.add("TikzGraphs")
Pkg.add("Documenter")
Pkg.add("Discretizers")
Pkg.add("RDatasets")
using Documenter, BayesNets, TikzGraphs, TikzPictures, Discretizers, RDatasets

makedocs(
    modules = [BayesNets, TikzPictures, TikzGraphs, Discretizers, RDatasets],
    format = Documenter.HTML(
        mathengine = Documenter.MathJax2()
    ),
    sitename = "BayesNets.jl",
    pages = [
        "Table of Contents" => [
            "index.md",
            "install.md",
            "usage.md",
            "concepts.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/sisl/BayesNets.jl.git",
)
return true
