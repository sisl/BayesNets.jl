
"""
Baysian Structure learning seeks to maximize P(G|D)
In the Bayesian fashion, we can provide a prior over the parameters in our learning network.
This is described using a Dirichlet Prior.
"""
abstract type DirichletPrior end

"""
A uniform Dirichlet prior such that all α are the same

Defaults to the popular K2 prior, α = 1, which is similar to Laplace Smoothing

    https://en.wikipedia.org/wiki/Additive_smoothing
"""
struct UniformPrior <: DirichletPrior
    α::Float64
    UniformPrior(α::Float64=1.0) = new(α)
end

Base.print(io::IO, p::UniformPrior) = Base.print(io, "UniformPrior(%.2f)", p.α)

Base.get(p::UniformPrior, ncategories::Integer) = fill(p.α, ncategories)
function Base.get(p::UniformPrior,
                  var_index::Integer,
                  ncategories::AbstractVector{I}, # [nvars]
                  parents::AbstractVector{J},    # [nvars]
                  ) where {I<:Integer, J<:Integer}

    r = ncategories[var_index]
    q = isempty(parents) ? 1 : prod(ncategories[parents])
    α = p.α

    fill(α, r, q)
end


"""
Assigns equal scores to Markov equivalent structures

    α_ijk = x/{q_i * r_i} for each j, k and some given x

see DMU section 2.4.3
"""
struct BDeuPrior <: DirichletPrior
    x::Float64
    BDeuPrior(x::Float64=1.0) = new(x)
end

Base.print(io::IO, p::BDeuPrior) = @printf(io, "BDeuPrior(%.2f)", p.x)

Base.get(p::BDeuPrior, ncategories::Integer) = fill(p.x/ncategories, ncategories)
function Base.get(p::BDeuPrior,
                  var_index::Integer,
                  ncategories::AbstractVector{I}, # [nvars]
                  parents::AbstractVector{J},    # [nvars]
                  ) where {I<:Integer, J<:Integer}

    x = p.x
    r = ncategories[var_index]
    q = isempty(parents) ? 1 : prod(ncategories[parents])

    α = x / (r*q)

    fill(α, r, q)
end
