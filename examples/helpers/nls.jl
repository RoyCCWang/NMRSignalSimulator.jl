# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

struct NNLSParameters{T <: AbstractFloat}
    lbs::Memory{T}
    #y::Memory{Complex{T}}

    function NNLSParameters(lbs::AbstractVector{T}) where T <: AbstractFloat
        for lb in lbs
            lb >= zero(T) || error("Entries in lbs must be non-negative.")
        end
        #return new{T}(Memory{T}(lbs), Memory{Complex{T}}(y))
        return new{T}(Memory{T}(lbs))
    end
end

struct NNLSState{T <: AbstractFloat}
    obs::Memory{T}

    function NNLSState(::Type{T}, N_fit_positions::Integer) where T <: AbstractFloat
        N_fit_positions > 0 || error("N_fit_positions must be positive.")
        
        c = Memory{T}(undef, 2*N_fit_positions)
        fill!(c, T(NaN))

        return new{T}(c)
    end
end

# B is input, y is parameter.
struct NNLSCallable end
function (nls!::NNLSCallable)(
    sol::AbstractVector,
    s::NNLSState{T},
    Br::AbstractMatrix{T},
    yr::AbstractVector{T},
    ps::NNLSParameters{T},
    ) where T <: AbstractFloat
    
    obs = s.obs
    lbs = ps.lbs
    
    # prepare NLS observation, subject to the lower bounds.
    mul!(obs, Br, lbs)
    obs .= yr .- obs

    x_star = vec(NNLS.nonneg_lsq(Br, obs)) # non-Gram option of running_lsq. Allocates.

    # add the lower bound back.
    sol .= x_star .+ lbs

    return nothing
end