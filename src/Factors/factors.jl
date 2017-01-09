#
# The whole submodule thing is weird, so ...
#

# TODO sub2ind for assignments
# TODO broadcast fallback for empty arrays
# TODO broadcast_reduce tag-team

include("factors_main.jl") # first because aux needs it
include("errors.jl")
include("auxiliary.jl")
include("factors_dims.jl")
include("factors_access.jl")
include("factors_dataframes.jl")
include("factors_io.jl")

