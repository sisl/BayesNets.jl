#
# Factors-DataFrames conversion
#
# Bridge between the two

"""
Convert a Factor to a DataFrame
"""
function Base.convert(::Type{DataFrame}, ft::Factor)
    df = DataFrames.DataFrame(pattern(ft))
    DataFrames.rename!(df, names(df), names(ft))
    df[:probability] = ft.probability[:]

    return df
end


