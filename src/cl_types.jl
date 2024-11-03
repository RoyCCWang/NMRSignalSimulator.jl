# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>


# # complex Lorentzian (CL).




# for one resonance group (i.e. one surrogate).
# disregard is λ0 is a negative value.
struct InterpolationSamples{T <: AbstractFloat}

    samples::Matrix{Complex{T}}

    r_min::T
    Δr::T
    r_max::T

    κ_λ_lb::T
    Δκ_λ::T
    κ_λ_ub::T

    hz_min::T
    hz_max::T
    λ0::T
end

# # default to invalid values. At deserialization, check value for λ0. If is negative, then we know it is the singlet case; no interpolation surrogates are used.
# function InterpolationSamples(::Type{T}) where T <: AbstractFloat
#     return InterpolationSamples(ones(Complex{T}, 1, 1), (-ones(T, 7))...)
# end

function InterpolationSamples(
    C::CLSurrogateConfig{T},
    s::Matrix{Complex{T}},
    r_min::T,
    r_max::T,
    u_min::T,
    u_max::T,
    λ0::T,
    ) where T <: AbstractFloat

    return InterpolationSamples(s, r_min, C.Δr, r_max, C.κ_λ_lb, C.Δκ_λ, C.κ_λ_ub, u_min, u_max, λ0)
end



### different parameterizations of the spin system FID parameters.

abstract type MoleculeParams end

# abstract type ShiftParms{T} <: MoleculeParms{T} end
# abstract type PhaseParms{T} <: MoleculeParms{T} end
# abstract type T2Parms{T} <: MoleculeParms{T} end

# struct CoherenceShift{T} <: SpinSysParams{T}
#     κs_λ::Vector{T} # a multiplier for each (spin group.
#     κs_β::Vector{Vector{T}} # a vector coefficient for each (spin group). vector length: number of spins in the spin group.
#     d::Vector{Vector{T}} # a multiplier for each (spin group, partition element).
#     κs_ζ::Vector{Vector{T}} # same size as κs_β. # intermediate buffer for d.
# end

# struct SharedParams <: T2Parms{T}
#     var::Vector{T} # multiplier wrt some λ0. length: number of spin groups.





# parameters for mixtures. won't work since Vector{abstract type}.
# SystemsTrait(::Type{<:Vector{Y}}) where Y <: MoleculeParams = Shared() # default for super type MoleculeParams.
# SystemsTrait(::Type{<:Vector{Y}}) where Y <: SharedParams = Shared()
# SystemsTrait(::Type{<:Vector{Y}}) where Y <: CoherenceParams = Coherence()

######## for construction a table for visualization and analysis.
abstract type TableConstructionTrait end
struct ProcessShifts <: TableConstructionTrait end
