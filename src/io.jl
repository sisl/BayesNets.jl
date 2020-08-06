#
# Bayes Net
#
lualatex_available() = try success(`lualatex -v`) catch; false end
Base.showable(::MIME"image/svg+xml", bn::BayesNet) = true


plot(dag::DAG, nodelabel) = gplot(dag,
								  nodelabel=nodelabel,
								  layout=stressmajorize_layout,
								  nodefillc="lightgray",
								  edgestrokec="black",
								  EDGELINEWIDTH=0.3) # GraphPlot (default plotting)


# called at runtime (replaces plot with TikzGraphs, if loaded)
function __init__()
	@require TikzGraphs="b4f28e30-c73f-5eaf-a395-8a9db949a742" begin
		if lualatex_available()
			plot(dag::DAG, nodelabel) = TikzGraphs.plot(dag, nodelabel)
		end
	end
end


function plot(bn::BayesNet)
	if !isempty(names(bn))
		dag = bn.dag
		nodelabel = AbstractString[string(s) for s in names(bn)] # NOTE: sometimes the same var shows up twice
	else
		dag = DiGraph(1)
		nodelabel = ["Empty Graph"]
	end

	plot(dag, nodelabel)
end


function Base.show(f::IO, a::MIME"image/svg+xml", bn::BayesNet)
 	show(f, a, plot(bn))
end
