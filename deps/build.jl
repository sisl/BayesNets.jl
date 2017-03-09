
try
    Pkg.clone("https://github.com/sisl/ProbabilisticGraphicalModels.jl.git")
catch e
    println("Exception when cloning $(url): $(e)")
end
