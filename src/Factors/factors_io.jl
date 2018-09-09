#
# Factors IO
#
# Printing IO stuff

# Base.show(io::IO, ϕ::Factor) = show(io, convert(DataFrame, ϕ))

function Base.show(io::IO, ϕ::Factor)
    print(io, "$(length(ϕ)) instantiations:")
    for (d, s) in zip(ϕ.dimensions, size(ϕ))
        println(io, "")
        print(io, "  $d ($s)")
    end
end

# # mimewriting doesn't use a monospace font, which I don't like ...
Base.showable(::MIME"text/html", ϕ::Factor) = true
Base.show(io::IO, a::MIME"text/html", ϕ::Factor) = show(io, a, convert(DataFrame, ϕ))
#    print(io, replace(replace(repr(ϕ), "\n", "<br>"), "\t",
#            "&emsp;&emsp;&emsp;"))

