# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

module NMRSignalSimulator

# Write your package code here.
using LinearAlgebra

#import Interpolations, OffsetArrays
import CubicBSplineInterpolation as ITP


import NMRHamiltonian # consider removing this dependency, or split the data structure into a separate package.

# inherit dependencies.
import JSON3


# constant values.
function twopi(::Type{T}) where T <: AbstractFloat
    return T(2*π)
end

function twopi(::Type{Float32})
    return 6.2831855f0
end

function twopi(::Type{Float64})
    return 6.283185307179586
end

# Float16 is emulated by Float32 as of Julia v1.10. We don't specialize on it.
# https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Floating-Point-Numbers

include("essentials/utils.jl")
include("essentials/surrogates.jl")
include("essentials/parameters.jl")
#include("cl_types.jl")


include("core/cl_engine.jl")
include("core/fid_engine.jl")
include("core/query.jl")
include("core/cost.jl")

include("reporting.jl")

include("core/derivatives.jl")

#T ODO public and export API in a future version.
export 

CLSurrogateConfig,
FIDSurrogateConfig,
create_cl_surrogate,
create_fid_surrogate,

create_rcs_list,
eval_fid,
eval_cl,

NLSCostState,

get_prelim_state,
get_num_compounds,
get_model_params
get_model_state
get_frequency_bounds
get_λ0,
get_Δc,
get_frequency_conversion,
get_num_groups,

NLSCostParameters,
NLSCostGradientCallable,
NLSCostGradientState

SystemT2,
initialize_params,
update_state!

end
