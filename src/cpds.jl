typealias Assignment Dict

module CPDs
abstract CPD

export CPD, Discrete, Bernoulli, Normal, domain

typealias NodeName Symbol

cpdDict(names::Vector{NodeName}, dict::Dict) = a -> dict[[a[n] for n in names]]

type Discrete <: CPD
  domain::AbstractArray{Any,1}
  parameterFunction::Function
  domainIndex::Dict{Any,Integer}
  function Discrete{T}(domain::AbstractArray{T,1}, parameterFunction::Function)
    new(domain, parameterFunction, Dict([domain[i] => i for i in 1:length(domain)]))
  end
  function Discrete{T,U}(domain::AbstractArray{T,1}, parameters::AbstractArray{U,1})
    new(domain, a->parameters, Dict([domain[i] => i for i in 1:length(domain)]))
  end
  function Discrete{T}(domain::AbstractArray{T,1}, names::AbstractArray{NodeName,1}, dict::Dict)
    new(domain, a->dict[[a[n] for n in names]], Dict([domain[i] => i for i in 1:length(domain)]))
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

domain(d::CPDs.Discrete) = DiscreteDomain(d.domain)
domain(d::CPDs.Bernoulli) = BinaryDomain()
domain(d::CPDs.Normal) = RealDomain()

function pdf(d::CPDs.Discrete, a::Assignment)
  (x) -> d.parameterFunction(a)[d.domainIndex[x]]
end

function pdf(d::CPDs.Bernoulli, a::Assignment)
  (x) -> x != 0 ? d.parameterFunction(a) : (1 - d.parameterFunction(a))
end

function rand(d::CPDs.Discrete, a::Assignment)
  p = d.parameterFunction(a)
  n = length(p)
  i = 1
  c = p[1]
  u = rand()
  while c < u && i < n
    c += p[i += 1]
  end
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
