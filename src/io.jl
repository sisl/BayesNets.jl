#
# Bayes Net
#
lualatex_available() = success(`lualatex -v`)
function Base.showable(::MIME"image/svg+xml", bn::BayesNet)
	try
		lualatex_available()
	catch
		true # LuaLaTeX not installed, use GraphPlot instead
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

	try
		if lualatex_available()
			plot(dag, nodelabel) # TikzGraphs
		else
			error("LuaLaTeX unsuccessful.") # lualatex installed, but did not return error code 0
		end
	catch err
		gplot(dag, nodelabel=nodelabel) # GraphPlot
	end
end


function Base.show(f::IO, a::MIME"image/svg+xml", bn::BayesNet)
 	show(f, a, plot(bn))
end
