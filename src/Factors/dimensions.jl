#
# Dimensions
#
# Main file for Dimension datatype

abstract Dimension{T}

"""
A dimension whose states emnumerated in an array. Order does not matter
"""
immutable CardinalDimension{T} <: Dimension{T}
    name::Symbol
    states::AbstractArray{T, 1}

    function CardinalDimension(name::Symbol, states::AbstractArray{T, 1})
        if !allunique(states)
            non_unique_states_error()
        end

        if length(states) < 2
            singleton_dimension_error(length(states))
        end

        new(name, states)
    end
end

CardinalDimension{T}(name::Symbol, states::AbstractArray{T, 1}) =
   CardinalDimension{T}(name, states)

abstract AbstractOrdinalDimension{T} <: Dimension{T}
abstract OrdinalRangeDimension{T} <: AbstractOrdinalDimension{T}

#support <, > comparisons, and an ordering in the states
immutable OrdinalDimension{T} <: AbstractOrdinalDimension{T}
    name::Symbol
    states::AbstractArray{T, 1}

    function OrdinalDimension(name::Symbol, states::AbstractArray{T, 1})
        if !allunique(states)
            non_unique_states_error()
        end

        if length(states) < 2
            singleton_dimension_error(length(states))
        end

        new(name, states)
    end
end

OrdinalDimension{T}(name::Symbol, states::AbstractArray{T, 1}) =
    OrdinalDimension{T}(name, states)

"""
Uses a Base.StepRange to enumerate possible states.

Starts at variable `start`, steps by `step`, ends at `stop` (or less,
if the math does't work out
"""
immutable OrdinalStepDimension{T, S} <: OrdinalRangeDimension{T}
    name::Symbol
    states::StepRange{T, S}

    function OrdinalStepDimension(name::Symbol, start::T, step::S, stop::T)
        states = start:step:stop

        if length(states) < 2
            singleton_dimension_error(length(states))
        end

        new(name, states)
    end
end

OrdinalStepDimension{T,S}(name::Symbol, start::T, step::S, stop::T) =
    OrdinalStepDimension{T,S}(name, start, step, stop)

OrdinalStepDimension{T}(name::Symbol, start::T, stop::T) =
    OrdinalStepDimension{T,Int}(name, start, 1, stop)

"""
Similar to a UnitRange, enumerates values over start:stop
"""
immutable OrdinalUnitDimension{T} <: OrdinalRangeDimension{T}
    name::Symbol
    states::UnitRange{T}

    function OrdinalUnitDimension(name::Symbol, start::T, stop::T)
        states = start:stop

        if length(states) < 2
            singleton_dimension_error(length(states))
        end

        new(name, states)
    end
end

OrdinalUnitDimension{T}(name::Symbol, start::T, stop::T) =
    OrdinalUnitDimension{T}(name, start, stop)

"""
An integer dimension that starts at 1

This is were the real magic happens, since all the optimizations will focus
on this one. If I optimize anything at all.
"""
immutable CartesianDimension{T<:Integer} <: OrdinalRangeDimension{T}
    name::Symbol
    states::Base.OneTo{T}

    CartesianDimension(name::Symbol, length::T) = length < convert(T, 2) ?
        singleton_dimension_error(length) : new(name, Base.OneTo(length))
end

CartesianDimension{T<:Integer}(name::Symbol, length::T) =
    CartesianDimension{T}(name, length)

###############################################################################
#                   Functions

@inline Base.length(d::Dimension) = length(d.states)
@inline Base.values(d::Dimension) = d.states
@inline name(d::Dimension) = d.name

"""
Get the first state in this dimension
"""
@inline function Base.first(d::Dimension)
    return first(d.states)
end

"""
Get the last state in this dimension
"""
@inline function Base.last(d::Dimension)
    return last(d.states)
end

"""
Return the data type of the dimension
"""
@inline function Base.eltype{T}(::Dimension{T})
    return T
end

###############################################################################
#                   Indexing

# custom indexin for integer (and float?) ranges
@inline function Base.indexin{T<:Integer}(xs::Vector{T}, d::OrdinalRangeDimension{T})
    inds = zeros(Int, size(xs))

    for (i, x) in enumerate(xs)
        if first(d.states) <= x <= last(d.states)
            (ind, rem) = divrem(x - first(d.states), step(d.states))
            if rem == 0
                inds[i] = ind + 1
            end
        end
    end

    return inds
end

@inline function Base.indexin{T<:Integer}(xs::Vector{T}, d::CartesianDimension{T})
    inds = zeros(Int, size(xs))

    for (i, x) in enumerate(xs)
        if first(d.states) <= x <= last(d.states)
            inds[i] = x
        end
    end

    return inds
end

@inline function Base.indexin{T}(x::T, d::Dimension{T})
    return first(indexin([x], d.states))
end

@inline function Base.indexin{T}(xs::Vector{T}, d::Dimension{T})
    return indexin(xs, d.states)
end

@inline function Base.indexin{T}(r::Range{T}, d::Dimension{T})
    return indexin(collect(r), d.states)
end

@inline function Base.indexin{T}(::Colon, d::Dimension{T})
    return collect(1:length(d))
end

###############################################################################
#                   Updating a Dimension

@inline function update(dim::CardinalDimension, I)
    return CardinalDimension(dim.name, dim.states[I])
end

@inline function update(dim::AbstractOrdinalDimension, I)
    return OrdinalDimension(dim.name, dim.states[I])
end

###############################################################################
#                   Comparisons and Equality

# ensure that cardinal and ordinal dimensions will always be unequal
@inline ==(::CardinalDimension, ::AbstractOrdinalDimension) = false

@inline ==(d1::CardinalDimension, d2::CardinalDimension) =
    (d1.name == d2.name) &&
    (length(d1.states) == length(d2.states)) &&
    all(d1.states .== d2.states)

@inline ==(d1::AbstractOrdinalDimension, d2::AbstractOrdinalDimension) =
    (d1.name == d2.name) &&
    (length(d1.states) == length(d2.states)) &&
    all(d1.states .== d2.states)

@inline function .=={T}(d::Dimension{T}, x)
    # states should all be unique
    return d.states .== convert(T, x)
end

@inline .!=(d::Dimension, x) = !(d .== x)

# can't be defined in terms of each b/c of
#  non-trivial case of x ∉ d
@inline function .<(d::AbstractOrdinalDimension, x)
    ind = d .==  x
    loc = findfirst(ind)

    # if x is in the array
    @inbounds if loc != 0
        ind[1:loc] = true
        ind[loc] = false
    end

    return ind
end

@inline function .>(d::AbstractOrdinalDimension, x)
    ind = d .== x
    loc = findfirst(ind)

    # if x is in the array
    @inbounds if loc != 0
        ind[loc:end] = true
        ind[loc] = false
    end

    return ind
end

@inline function .<=(d::AbstractOrdinalDimension, x)
    ind = d .== x
    loc = findfirst(ind)
    @inbounds ind[1:loc] = true

    return ind
end

@inline function .>=(d::AbstractOrdinalDimension, x)
    ind = d .== x
    loc = findfirst(ind)
    loc == 0 || @inbounds ind[loc:end] = true

    return ind
end

@inline function Base.in(x, d::Dimension)
    return any(d .== x)
end

###############################################################################
#                   IO Stuff

@inline Base.mimewritable(::MIME"text/html", d::Dimension) = true

@inline Base.show(io::IO, d::Dimension) =
    print(io, "$(d.name): $(repr(d.states)) ($(length(d)))")
@inline Base.show(io::IO, a::MIME"text/html", d::Dimension) = show(io, d)

@inline Base.show(io::IO, d::CartesianDimension) =
    print(io, "$(d.name): 1:$(last(d))")
@inline Base.show(io::IO, a::MIME"text/html", d::CartesianDimension) = show(io, d)

