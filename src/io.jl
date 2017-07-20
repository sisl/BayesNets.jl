Base.mimewritable(::MIME"image/svg+xml", bn::BayesNet) = success(`lualatex -v`)
Base.mimewritable(::MIME"text/html", dfs::Vector{DataFrame}) = true

function plot(bn::BayesNet)
	if !isempty(names(bn))
		plot(bn.dag, AbstractString[string(s) for s in names(bn)]) # NOTE: sometimes the same var shows up twice
	else
		plot(DiGraph(1), ["Empty Graph"])
	end
end

function Base.show(f::IO, a::MIME"image/svg+xml", bn::BayesNet)
 	show(f, a, plot(bn))
end

function Base.show(io::IO, a::MIME"text/html", dfs::Vector{DataFrame})
	for df in dfs
		writemime(io, a, df)
	end
end
