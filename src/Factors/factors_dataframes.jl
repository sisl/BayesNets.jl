#
# Factors-DataFrames conversion
#
# Bridge between the two

"""
Convert a Factor to a DataFrame
"""
function Base.convert(::Type{DataFrame}, ϕ::Factor)
    df = DataFrames.DataFrame(pattern(ϕ))
    DataFrames.rename!(df, [f => t for (f, t) = zip(names(df), names(ϕ))] )
    df[:potential] = ϕ.potential[:]

    return df
end


function DataFrame(ϕ::Factor)
    return convert(DataFrame, ϕ)
end
