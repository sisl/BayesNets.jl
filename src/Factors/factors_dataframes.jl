#
# Factors-DataFrames
#
# Bridge between the two

"""
Convert a Factor to a DataFrame
"""
function DataFrames.DataFrame(ft::Factor)
    df = DataFrames.DataFrame(pattern(ft))
    DataFrames.rename!(df, names(df), names(ft))
    df[:probability] = ft.probability[:]

    return df
end


