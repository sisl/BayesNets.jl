using Documenter, BayesNets, TikzGraphs, TikzPictures, Discretizers, RDatasets

page_order = [
    "index.md",
    "install.md",
    "usage.md",
    "concepts.md",
    "api.md"
]

makedocs(
    modules = [BayesNets],
    format = Documenter.HTML(
        mathengine = Documenter.MathJax2(),
        size_threshold_ignore=["api.md"]
    ),
    sitename = "BayesNets.jl",
    pages = [
        "Table of Contents" => page_order
    ],
    warnonly = [:missing_docs]
)

deploydocs(
    repo = "github.com/sisl/BayesNets.jl.git",
    push_preview=true
)
