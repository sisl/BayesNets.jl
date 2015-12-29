abstract Domain

type DiscreteDomain <: Domain
  elements::Vector
end

type ContinuousDomain <: Domain
  lower::Real
  upper::Real
end

BinaryDomain() = DiscreteDomain([false, true])
RealDomain() = ContinuousDomain(-Inf, Inf)