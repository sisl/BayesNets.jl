#=
A CPD is a Conditional Probability Table
It represents P(X|Y)
=#

module CPDs

export CPD, DiscreteCPD, DiscreteFunctionCPD, DiscreteDictCPD, DiscreteStaticCPD, BernoulliCPD, NormalCPD
export Domain, DiscreteDomain, ContinuousDomain, BINARY_DOMAIN, REAL_DOMAIN
export domain, probvec, pdf
export Assignment, NodeName

typealias Assignment Dict
typealias NodeName Symbol

"""
helper function for sampling from a vector of probabilities
Returns the sampled item where the probabilty of i is p[i]
"""
function Base.rand(p::AbstractVector{Float64})
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end
    i
end

# cpdDict(names::Vector{NodeName}, dict::Dict) = a -> dict[[a[n] for n in names]]

"""
A domain defines the types and values of variables that the variables
in a CPD are defined over
"""
abstract Domain

type DiscreteDomain <: Domain
    elements::Vector
end

type ContinuousDomain <: Domain
    lower::Real
    upper::Real
end

const BINARY_DOMAIN = DiscreteDomain([false, true])
const REAL_DOMAIN = ContinuousDomain(-Inf, Inf)
"""
Abstract CPD type
Each CPD should implement:

    domain(CPD) → ::Domain
    pdf(CPD, ::Assignment) → (Function(x) → p(x)::Float64)
    Base.rand(CPD, ::Assignment) → an element from the domain selected according to P(X)
    if discrete: probvec(CPD, ::Assignment) → Vector{Float64} of probabilities

"""
abstract CPD

abstract DiscreteCPD <: CPD
domain(d::DiscreteCPD) = DiscreteDomain(d.domain)
function pdf(d::DiscreteCPD, a::Assignment)
    (x) -> probvec(d)[d.domainIndex[x]]
end
function Base.rand(d::DiscreteCPD, a::Assignment)
    p = probvec(d, a)
    i = rand(p)
    d.domain[i]
end

"""
A CPD in which P(x|y) is over discrete values, defined using custom function
"""
type DiscreteFunctionCPD <: DiscreteCPD
    domain::AbstractVector{Any} # set of discrete values the CPD covers
    parameterFunction::Function
    domainIndex::Dict{Any,Integer}

    DiscreteFunctionCPD{T}(domain::AbstractVector{T}, parameterFunction::Function) =
        new(domain, parameterFunction, Dict([domain[i] => i for i in 1:length(domain)]))
    DiscreteFunctionCPD{T,U}(domain::AbstractVector{T}, parameters::AbstractVector{U}) =
        new(domain, a->parameters, Dict([domain[i] => i for i in 1:length(domain)]))
    DiscreteFunctionCPD{T}(domain::AbstractVector{T}, names::AbstractVector{NodeName}, dict::Dict) =
        new(domain, a->dict[[a[n] for n in names]], Dict([domain[i] => i for i in 1:length(domain)]))
end
probvec(d::DiscreteFunctionCPD, a::Assignment) = d.parameterFunction(a)

"""
A CPD in which P(x|y) is over discrete values, defined using a dictionary
"""
type DiscreteDictCPD <: DiscreteCPD
  
    domain::AbstractVector{Any}
    keys::Vector{Symbol}
    probabilitylookup::Dict{Dict, Vector{Float64}} # NOTE(tim): so we don't have functions that cannot be saved by JLD
    domainIndex::Dict{Any,Int}

    function DiscreteDictCPD{T, D<:Dict}(domain::AbstractVector{T}, probabilitylookup::Dict{D, Vector{Float64}})
        keyset = collect(keys(first(probabilitylookup)[1]))
        new(domain, keyset, probabilitylookup, Dict([domain[i] => i for i in 1:length(domain)]))
    end
end
function probvec(d::DiscreteDictCPD, a::Assignment)
    lookup = Dict{Symbol,Any}()
    for sym in d.keys
        lookup[sym] = a[sym]
    end
    d.probabilitylookup[lookup]
end

"""
A CPD in which P(x|y) is over discrete values and never changes
"""
type DiscreteStaticCPD <: DiscreteCPD
    
    domain::AbstractVector{Any}
    probs::Vector{Float64}
    domainIndex::Dict{Any,Int}
    
    DiscreteStaticCPD{T}(domain::AbstractVector{T}, probs::AbstractVector{Float64}) =
        new(domain, probs, Dict([domain[i] => i for i in 1:length(domain)]))
end
probvec(d::DiscreteStaticCPD, a::Assignment) = d.probs

"""
A CPD in which P(x|y) is a Bernoulli distribution
"""
type BernoulliCPD <: CPD
    parameterFunction::Function # a → P(x = true | a)
    BernoulliCPD(parameter::Real = 0.5) = new(a->parameter)
    BernoulliCPD(parameterFunction::Function) = new(parameterFunction)
    BernoulliCPD(names::AbstractVector{NodeName}, dict::Dict) = new(a->dict[[a[n] for n in names]])
end
domain(d::BernoulliCPD) = BINARY_DOMAIN
probvec(d::BernoulliCPD, a::Assignment) = [d.parameterFunction(a), 1.0-d.parameterFunction(a)]
function pdf(d::BernoulliCPD, a::Assignment)
    (x) -> x != 0 ? d.parameterFunction(a) : (1 - d.parameterFunction(a))
end
function Base.rand(d::BernoulliCPD, a::Assignment)
    rand() < d.parameterFunction(a)
end

"""
A CPD in which P(x|y) is a Gaussian
"""
type NormalCPD <: CPD
    parameterFunction::Function # a → (μ, σ)
    NormalCPD(parameterFunction::Function) = new(parameterFunction)
    NormalCPD(mu::Real, sigma::Real) = new(a->(mu, sigma))
end
domain(d::NormalCPD) = REAL_DOMAIN
function pdf(d::NormalCPD, a::Assignment)
    (mu::Float64, sigma::Float64) = d.parameterFunction(a)
    x -> begin x
        z = (x - mu)/sigma
        exp(-0.5*z*z)/(√2π*sigma)
    end
end
function Base.rand(d::CPDs.NormalCPD, a::Assignment)
    mu, sigma = d.parameterFunction(a)::Tuple{Float64, Float64}
    mu + randn() * sigma
end

end # module CPDs