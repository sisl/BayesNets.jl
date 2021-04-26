push!(LOAD_PATH, "../src")
import Pkg
Pkg.add("BayesNets")
using Documenter, BayesNets

makedocs(
    modules = [BayesNets],
    format = Documenter.HTML(),
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