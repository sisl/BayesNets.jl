#
# Factors-DataFrames conversion
#
# Bridge between the two

"""
Convert a Factor to a DataFrame
"""
function Base.convert(::Type{DataFrame}, ϕ::Factor)
    df = DataFrames.DataFrame(pattern(ϕ))
    DataFrames.rename!(df, names(df), names(ϕ))
    df[:potential] = ϕ.potential[:]

    return df
end


