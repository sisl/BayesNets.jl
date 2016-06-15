"""
TODO: rename table() to factor()?
Constructs the CPD factor associated with the given node in the BayesNet
"""
function table(bn::BayesNet, name::NodeName)

  d = DataFrame()
    c = cpd(bn, name)
    names = push!(parents(bn, name), name)

    nparents = length(names)-1
    if nparents > 0
        A = ndgrid([domain(bn, name).elements for name in names]...)
        for (i,name2) in enumerate(names)
            d[name2] = vec(A[i])
        end
    else
        d[name] = domain(bn, name).elements
    end

    p = ones(size(d,1)) # the probability column
    for i in 1:size(d,1)
        ownValue = d[i,length(names)]
        assignment = Dict([names[j]=>d[i,j] for j in 1:nparents])
        pdf_func = pdf(c, assignment)
        p[i] = pdf_func(ownValue)
    end
    d[:p] = p
    d
end

table(bn::BayesNet, name::NodeName, a::Assignment) = select(table(bn, name), a)