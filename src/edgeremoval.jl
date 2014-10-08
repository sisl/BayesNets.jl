import Graphs: target, source, edge_index, vertex_index

# patch from https://github.com/JuliaLang/Graphs.jl/pull/86/files
function remove_edge!{V,E}(g::GenericGraph{V,E}, u::V, v::V, e::E)
  @assert e in g.edges && source(e, g) == u && target(e, g) == v
  ei = edge_index(e, g)::Int
  ui = vertex_index(u, g)::Int
  vi = vertex_index(v, g)::Int

  for i = 1:length(g.finclist[ui])
    if g.finclist[ui][i] == e
      splice!(g.finclist[ui], i)
      break
    end # if
  end # for

  for j = 1:length(g.binclist[vi])
    if g.binclist[vi][j] == e
      splice!(g.binclist[vi], j)
      break
    end # if
  end # for

  splice!(g.edges, ei)

  if !g.is_directed
    rev_e = revedge(e)
    for i = 1:length(g.finclist[vi])
      if g.finclist[vi][i] == rev_e
        splice!(g.finclist[vi], i)
        break
      end # if
    end # for

    for j = 1:length(g.binclist[ui])
      if g.binclist[ui][j] == rev_e
        splice!(g.binclist[ui], j)
        break
      end # if
    end # for
  end # if
end


# Needed since edge indexing is not unique. That is, if e = edge(1, 2) is in graph g, then e != make_edge(g, 1, 2).
function remove_edge!{V,E}(g::GenericGraph{V,E}, u::V, v::V)
  for edge in g.edges
    if source(edge, g) == u && target(edge, g) == v
      remove_edge!(g, u, v, edge)
      break
    end # if
  end #for
end

remove_edge!{V,E}(g::GenericGraph{V,E}, e::E) = remove_edge!(g, source(e, g), target(e, g))