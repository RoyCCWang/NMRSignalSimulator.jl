# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>
# SPDX-License-Identifier: GPL-3.0-only



struct FrequencyConversion{T <: AbstractFloat}
    ν_0ppm::T
    SW::T
    fs::T
end

function ppm_to_hz(p::Real, A::FrequencyConversion)
    return A.ν_0ppm + p*A.fs/A.SW
end

function hz_to_ppm(u::Real, A::FrequencyConversion)
    return (u - A.ν_0ppm)*A.SW/A.fs
end

function FrequencyConversion(A::NMRHamiltonian.SHType)
    return FrequencyConversion(A.ν_0ppm, A.SW, A.fs)
end

# ppm2radfunc = pp->ppm_to_hzfunc(pp)*2*π
function ppm_to_rad(p::T, A::FrequencyConversion) where T
    return ppm_to_hz(p, A)*T(2*π)
end

# rad2ppmfunc = ww->hz2ppmfunc(ww/(2*π))
function rad_to_ppm(ω::T, A::FrequencyConversion) where T
    return hz_to_ppm(ω/T(2*π), A)
end

function Δppm_to_Δrad(Δp::T, A::FrequencyConversion) where T
    return ppm_to_rad(Δp, A) - ppm_to_rad(zero(T), A)
end

# inverse of Δppm2Δrad().
function Δrad_to_Δppm(Δω::T, A::FrequencyConversion) where T
    return rad_to_ppm(Δω + ppm_to_rad(zero(T), A), A)
end

# function get_group_vs_system_mapping(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat
    
    
#     LUT = Memory{Tuple{Int,Int,Int}}(undef, NMRHamiltonian.get_num_groups(As))

#     sys_ind = 0
#     st_ind = 0
#     fin_ind = 0
#     for n in eachindex(As)
#         for i in As[n].Δc_bar

#             st_ind += 1
#             fin_ind = st_ind + length(As[n].Δc_bar[i]) - 1
            
#             sys_ind += 1
#             #LUT[sys_ind] = 
#         end
#     end
#     return LUT
# end