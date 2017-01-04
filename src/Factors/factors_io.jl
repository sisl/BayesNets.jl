#
# Factors IO
#
# Printing IO stuff

function Base.show(io::IO, ft::Factor)
    print(io, "$(length(ft)) instantiations:")
    for (d, s) in zip(ft.dimensions, size(ft))
        println(io, "")
        print(io, "  $d ($s)")
    end
end

# mimewriting doesn't use a monospace font, which I don't like ...
#Base.mimewritable(::MIME"text/html", ft::Factor) = true
#Base.show(io::IO, a::MIME"text/html", ft::Factor) =
#    print(io, replace(replace(repr(ft), "\n", "<br>"), "\t",
#            "&emsp;&emsp;&emsp;"))

