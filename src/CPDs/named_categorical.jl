const MapableTypes = Union{AbstractString, Symbol}
struct NamedCategorical{N<:MapableTypes} <: DiscreteUnivariateDistribution
    cat::Categorical
    map::CategoricalDiscretizer{N, Int}
end
function NamedCategorical(items::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    cat = Categorical(probs./sum(probs))
    map = CategoricalDiscretizer(items)
    NamedCategorical{N}(cat, map)
end

function Base.show(io::IO, d::NamedCategorical)
    println(io, "NamedCategorical with entries:")
    for val in keys(d.map.n2d)
        @printf(io, "\t%8.4f:  %s\n", pdf(d, val), string(val))
    end
end

Distributions.ncategories(d::NamedCategorical) = Distributions.ncategories(d.cat)
Distributions.probs(d::NamedCategorical) = Distributions.probs(d.cat)
Distributions.params(d::NamedCategorical) = Distributions.params(d.cat)

Distributions.pdf(d::NamedCategorical{N}, x::N) where {N<:MapableTypes} = Distributions.pdf(d.cat, encode(d.map, x))
Distributions.logpdf(d::NamedCategorical{N}, x::N) where {N<:MapableTypes} = Distributions.logpdf(d.cat, encode(d.map, x))
Base.rand(d::NamedCategorical) = rand(sampler(d))

struct MappedAliasTable <: Sampleable{Univariate,Discrete}
   alias::Distributions.AliasTable
   map::CategoricalDiscretizer
end
Distributions.ncategories(s::MappedAliasTable) = Distributions.ncategories(s.alias)
Random.rand(s::MappedAliasTable) = decode(s.map, rand(s.alias))
Base.show(io::IO, s::MappedAliasTable) = @printf(io, "MappedAliasTable with %d entries", ncategories(s))

Distributions.sampler(d::NamedCategorical) = MappedAliasTable(Distributions.sampler(d.cat), d.map)
