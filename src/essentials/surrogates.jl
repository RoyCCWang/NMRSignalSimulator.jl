# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# helpers

function create_Δc_bar_flat(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat

    dc_bars = Memory{Memory{T}}(undef, NMRHamiltonian.get_num_groups(As))
    LUT = get_rg_flat_mapping(As)
    
    l = 0
    for (n,i,k) in LUT
        
        l += 1
        dc_bars[l] = Memory{T}(As[n].Δc_bar[i][k])
    end

    return dc_bars
end

function get_rg_flat_mapping(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat
    
    LUT = Memory{Tuple{Int,Int,Int}}(undef, NMRHamiltonian.get_num_groups(As))

    l = 0
    for n in eachindex(As)
        for i in eachindex(As[n].Δc_bar)
            for k in eachindex(As[n].Δc_bar[i])
                l += 1
                LUT[l] = (n,i,k)
            end
        end
    end
    return LUT
end

function create_Δc_bar_matrix(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat

    nested = Memory{Memory{Matrix{T}}}(undef, length(As))

    for n in eachindex(As, nested)
        nested[n] = Memory{Matrix{T}}(undef, length(As[n].Δc_bar))

        for i in eachindex(As[n].Δc_bar, nested[n])
            dc = As[n].Δc_bar[i]

            K = length(dc)
            D = length(dc[begin]) # they should all have the same dim.

            nested[n][i] = Matrix{T}(undef, K, D)

            for k in eachindex(dc)
                v = view(nested[n][i], k, :)
                copy!(v, dc[k])
            end
        end
    end
    return nested
end

# # # Model specification

struct ResonanceGroupDCs{T <: AbstractFloat}
    
    flat::Memory{Memory{T}} # [flatten group index][ME nuclei index]
    nested::Memory{Memory{Matrix{T}}} # [compound][spin systme][group_index, ME nuclei index]

    function ResonanceGroupDCs(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat
        return new{T}(create_Δc_bar_flat(As), create_Δc_bar_matrix(As))
    end
end

function get_nested(A::ResonanceGroupDCs)
    return A.nested
end

function get_flat(A::ResonanceGroupDCs)
    return A.flat
end

function get_num_groups(cbs::ResonanceGroupDCs)
    return length(cbs.flat)
end

function get_num_sys(cbs::ResonanceGroupDCs)
    
    N_sys = 0
    for cb_n in cbs.nested
        N_sys += length(cb_n)
    end
    return N_sys
end

abstract type SurrogateModel{T} end
# Interface requirements: have the field `qs`

function get_proxies(model::SurrogateModel)
    return model.qs
end

function get_num_compounds(model::SurrogateModel)
    return length(get_proxies(model))
end

struct FIDSurrogateModel{T <: AbstractFloat} <: SurrogateModel{T} # parameters for surrogate model.

    # one per resonance group.
    qs::Memory{Memory{Memory{ITP.Interpolator1DComplex{T}}}}

    Δc_bars::ResonanceGroupDCs{T}
    
    # base T2. formula: λ = ξ*λ0.
    λ0::T
    frequency_conversion::FrequencyConversion{T}
end

# This is without the compensation amplitude parameter, κ_α, denoted κs_α in code.
struct CLSurrogateModel{T <: AbstractFloat} <: SurrogateModel{T} # parameters for surrogate model.

    # one per resonance group.
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}}

    Δc_bars::ResonanceGroupDCs{T}
    
    # base T2. formula: λ = ξ*λ0.
    λ0::T
    frequency_conversion::FrequencyConversion{T}
end

function get_λ0(model::SurrogateModel)
    return model.λ0
end

function get_Δc(model::SurrogateModel)
    return model.Δc_bars
end

function get_frequency_conversion(model::SurrogateModel)
    return model.frequency_conversion
end

function get_num_groups(model::SurrogateModel)
    return get_num_groups(model.Δc_bars)
end

# note: fs/SW = spectrometer freq in MHz.
function get_hz_per_ppm(model::SurrogateModel{T}) where T <: AbstractFloat
    b = ppm_to_hz(one(T), model.frequency_conversion)
    a = ppm_to_hz(zero(T), model.frequency_conversion)
    return b - a
end

abstract type SurrogateInfo{T} end
# Interface requirements: must have field members: `group_LUT` and `rcs`.
# must have the methods `get_rcs`, and `get_group_LUT`.

function get_rcs(A::SurrogateInfo)
    return A.rcs
end

function get_group_LUT(A::SurrogateInfo)
    return A.group_LUT
end

struct CLSurrogateInfo{T} <: SurrogateInfo{T}
    hz_lb::T
    hz_ub::T
    group_LUT::Memory{Tuple{Int,Int,Int}}
    rcs::Memory{Memory{Memory{T}}} # inner-most length is N_nuclei for the spin sys..
end

# Units in Hz.
function get_frequency_bounds(A::CLSurrogateInfo)
    return A.hz_lb, A.hz_ub
end

struct FIDSurrogateInfo{T} <: SurrogateInfo{T}
    t_lb::T
    t_ub::T
    group_LUT::Memory{Tuple{Int,Int,Int}}
    rcs::Memory{Memory{Memory{T}}}
end

# units in seconds.
function get_time_bounds(A::CLSurrogateInfo)
    return A.t_lb, A.t_ub
end

######## # State specification

struct RGState{T <: AbstractFloat}
    ζ::T # in radians per second.
    cis_β::Complex{T} # in radians.
end

function default_RGState(::Type{T}) where T <: AbstractFloat
    return RGState(zero(T), cis(zero(T))) 
end

# # Complex lorentzian (CL) model
abstract type SurrogateCache{T} end

struct SystemT2Cache{T <: AbstractFloat} <: SurrogateCache{T}
    rg::Memory{Memory{Memory{RGState{T}}}}
    λs::Memory{Memory{T}}

    # for doing the dot product between parameters and Δc_bar, for each (n,i) spin system.
    ζ_buffer::Memory{Memory{Memory{T}}} # inner-most length is N_groups for the spin sys.
    β_buffer::Memory{Memory{Memory{T}}} # inner-most length is N_groups for the spin sys.

    function SystemT2Cache(model::SurrogateModel{T}, info::SurrogateInfo{T}) where T <: AbstractFloat
        cbs = model.Δc_bars
        λ0 = model.λ0
        rcs = get_rcs(info)        

        rg_counter = 0
        sys_counter = 0
        
        #N_groups = get_num_groups(cbs)
        N_compounds = length(get_nested(cbs))
        λs = Memory{Memory{T}}(undef, N_compounds)
        #rg_flat = Memory{RGState{T}}(undef, N_groups)
        rg_nested = Memory{Memory{Memory{RGState{T}}}}(undef, N_compounds)
        ζ_buffer = deepcopy(rcs)
        β_buffer = deepcopy(rcs)

        for n in eachindex(cbs.nested)
    
            N_sys = length(cbs.nested[n])
            rg_nested[n] = Memory{Memory{RGState{T}}}(undef, N_sys)

            λs[n] = Memory{T}(undef, N_sys)
            fill!(λs[n], λ0) # initialize T2 to λ0.
    
            for i in eachindex(cbs.nested[n])
                sys_counter += 1
    
                N_groups = size(cbs.nested[n][i], 1)
                rg_nested[n][i] = Memory{RGState{T}}(undef, N_groups)
    
                for k in axes(cbs.nested[n][i], 1)
                    rg_counter += 1
    
                    rg_nested[n][i][k] = default_RGState(T)
                end

                ζ_buffer[n][i] = Memory{T}(undef, N_groups)
                β_buffer[n][i] = Memory{T}(undef, N_groups)
            end
        end
    
        return new{T}(rg_nested, λs, ζ_buffer, β_buffer)#, ws)
    end
end

# For testing.
function flatten_ζ(state::SystemT2Cache)
    return collect(
        Iterators.flatten(
            Iterators.flatten(
                Iterators.flatten(
                    state.rg[n][i][k].ζ for k in eachindex(state.rg[n][i])
                ) for i in eachindex(state.rg[n])
            ) for n in eachindex(state.rg)      
        )
    )
end

# For testing.
function flatten_cis_β(state::SystemT2Cache)
    return collect(
        Iterators.flatten(
            Iterators.flatten(
                Iterators.flatten(
                    state.rg[n][i][k].cis_β for k in eachindex(state.rg[n][i])
                ) for i in eachindex(state.rg[n])
            ) for n in eachindex(state.rg)      
        )
    )
end

