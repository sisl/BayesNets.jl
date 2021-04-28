push!(LOAD_PATH, "../src")
import Pkg
Pkg.add("BayesNets")
Pkg.add("Documenter")
using Documenter, BayesNets

makedocs(
    modules = [BayesNets],
    format = Documenter.HTML(
        mathengine = Documenter.MathJax()
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
    repo = "github.com/dwijenchawra/BayesNets.jl.git",
)