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

#Base.mimewritable(::MIME"text/html", ft::Factor) = true
#Base.show(io::IO, a::MIME"text/html", ft::Factor) =
#    print(io, replace(replace(repr(ft), "\n", "<br>"), "\t",
#            "&emsp;&emsp;&emsp;"))

