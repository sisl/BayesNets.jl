module CPDs
abstract CPD

export CPD, DiscreteCPD, DiscreteFunctionCPD, DiscreteDictCPD, DiscreteStaticCPD, Bernoulli, Normaldomain

typealias Assignment Dict # TODO(tim): these are declared in BayesNets.jl as well
typealias NodeName Symbol

cpdDict(names::Vector{NodeName}, dict::Dict) = a -> dict[[a[n] for n in names]]

abstract DiscreteCPD <: CPD
type DiscreteFunctionCPD <: DiscreteCPD
  domain::AbstractArray{Any,1}
  parameterFunction::Function
  domainIndex::Dict{Any,Integer}
  function DiscreteFunctionCPD{T}(domain::AbstractArray{T,1}, parameterFunction::Function)
    new(domain, parameterFunction, [domain[i] => i for i in 1:length(domain)])
  end
  function DiscreteFunctionCPD{T,U}(domain::AbstractArray{T,1}, parameters::AbstractArray{U,1})
    new(domain, a->parameters, [domain[i] => i for i in 1:length(domain)])
  end
  function DiscreteFunctionCPD{T}(domain::AbstractArray{T,1}, names::AbstractArray{NodeName,1}, dict::Dict)
    new(domain, a->dict[[a[n] for n in names]], [domain[i] => i for i in 1:length(domain)])
  end
end
type DiscreteDictCPD <: DiscreteCPD
  domain::AbstractArray{Any,1}
  keys::Vector{Symbol}
  probabilitylookup::Dict{Dict, Vector{Float64}} # NOTE(tim): so we don't have functions that cannot be saved by JLD
  domainIndex::Dict{Any,Int}
  function DiscreteDictCPD{T, D<:Dict}(domain::AbstractArray{T,1}, probabilitylookup::Dict{D, Vector{Float64}})
    keyset = collect(keys(first(probabilitylookup)[1]))
    new(domain, keyset, probabilitylookup, [domain[i] => i for i in 1:length(domain)])
  end
end
type DiscreteStaticCPD <: DiscreteCPD
  domain::AbstractArray{Any,1}
  probs::Vector{Float64}
  domainIndex::Dict{Any,Int}
  function DiscreteStaticCPD{T}(domain::AbstractArray{T,1}, probs::AbstractVector{Float64})
    new(domain, probs, [domain[i] => i for i in 1:length(domain)])
  end
end

type Bernoulli <: CPD
  parameterFunction::Function
  function Bernoulli(parameterFunction::Function)
    new(parameterFunction)
  end
  function Bernoulli(parameter::Real = 0.5)
    new(a->parameter)
  end
  function Bernoulli(names::AbstractArray{NodeName,1}, dict::Dict)
    new(a->dict[[a[n] for n in names]])
  end
end

type Normal <: CPD
  parameterFunction::Function
  function Normal(parameterFunction::Function)
    new(parameterFunction)
  end
  function Normal(mu::Real, sigma::Real)
    new(a->[mu, sigma])
  end
end

end # module CPDs


########################################################
#
#
#
########################################################

# TODO(tim): can this be restructured to be in the CPDs module?

domain(d::CPDs.DiscreteCPD) = DiscreteDomain(d.domain)
domain(d::CPDs.Bernoulli) = BinaryDomain()
domain(d::CPDs.Normal) = RealDomain()

probvec(d::CPDs.DiscreteFunctionCPD, a::Assignment) = d.parameterFunction(a)
probvec(d::CPDs.DiscreteStaticCPD, a::Assignment) = d.probs
function probvec(d::CPDs.DiscreteDictCPD, a::Assignment)
  lookup = Dict{Symbol,Any}()
  for sym in d.keys
    lookup[sym] = a[sym]
  end
  d.probabilitylookup[lookup]
end

function pdf(d::CPDs.DiscreteCPD, a::Assignment)
  (x) -> probvec(d)[d.domainIndex[x]]
end

function pdf(d::CPDs.Bernoulli, a::Assignment)
  (x) -> x != 0 ? d.parameterFunction(a) : (1 - d.parameterFunction(a))
end

function rand(p::AbstractVector{Float64})
  n = length(p)
  i = 1
  c = p[1]
  u = rand()
  while c < u && i < n
    c += p[i += 1]
  end
  i
end
function rand(d::CPDs.DiscreteCPD, a::Assignment)
  p = probvec(d, a)
  i = rand(p)
  return d.domain[i]
end

function rand(d::CPDs.Bernoulli, a::Assignment)
  rand() < d.parameterFunction(a)
end

function pdf(d::CPDs.Normal, a::Assignment)
  (mu::Float64, sigma::Float64) = d.parameterFunction(a)
  function f(x)
    z = (x - mu)/sigma
    exp(-0.5*z*z)/(√2π*sigma)
  end
end

function rand(d::CPDs.Normal, a::Assignment)
  (mu::Float64, sigma::Float64) = d.parameterFunction(a)
  mu + randn() * sigma
end