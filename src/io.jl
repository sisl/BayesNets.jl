#
# Bayes Net
#
Base.showable(::MIME"image/svg+xml", bn::BayesNet) = success(`lualatex -v`)

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
