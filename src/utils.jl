# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>


"""
```
function Δcs2ζ(Δcs::T, ppm2hzfunc)::T where T
```

Convert ppm to radial frequency.
'''
# # test.
a = Δcs2ζ(0.1, ppm2hzfunc)
@show ζ2Δcs(a, ν_0ppm, hz2ppmfunc)
'''
"""
function Δcs2ζ(Δcs::T, ppm2hzfunc)::T where T

    return (ppm2hzfunc(Δcs)-ppm2hzfunc(zero(T)))*2*π
end

"""
```
function ζ2Δcs(ζ::T, ν_0ppm::T, hz2ppmfunc)::T where T
```

Convert radial frequency to ppm.
'''
# # test.
a = Δcs2ζ(0.1, ppm2hzfunc)
@show ζ2Δcs(a, ν_0ppm, hz2ppmfunc)
'''
"""
function ζ2Δcs(ζ::T, ν_0ppm::T, hz2ppmfunc)::T where T
    return hz2ppmfunc(ζ/twopi(T) + ν_0ppm)
end


"""
```
function fetchbounds(
    p::MixtureModelParameters,
    Bs::Vector{MoleculeType{T, SST, OT}};
    shift_proportion = 0.9, # between 0 to 1. Control the returned shift bounds as a proportion of the maximum allowed shift bounds used when the surrogates were created.
    phase_lb = convert(T, -π),
    phase_ub = convert(T, π),
)::Tuple{Vector{T},Vector{T}} where {T,SST,OT}
```

Returns a `Vector` for lower bound and `Vector` for upper bound for each parameter variable. The order (first elements to last elements) of the returned vectors are: Shift (ζ), phase (κ_β), then T2 (κ_λ) parameters.

Input:

- `p` -- model parameters. Used to determine the size of the output.

- `Bs` -- Surrogate model. This function uses the `CLOperationRange` field from each element of `Bs`.

Optional inputs:

- `shift_proportion` -- between 0 to 1. Control the returned shift bounds as a proportion of the maximum allowed shift bounds used when the surrogates were created.

- `phase_lb` -- the fill value for the lower bound of phase parameter (κ_β).

- `phase_ub` -- the fill value for the lower bound of phase parameter (κ_β).

"""
function fetchbounds(
    p::MixtureModelParameters,
    Bs::Vector{MoleculeType{T, SST, OT}};
    shift_proportion = 0.9,
    phase_lb = convert(T, -π),
    phase_ub = convert(T, π),
    )::Tuple{Vector{T},Vector{T}} where {T,SST, OT}

    @assert zero(T) < shift_proportion < one(T)

    mapping = p.systems_mapping

    lbs = Vector{T}(undef, length(p.var_flat))
    ubs = Vector{T}(undef, length(p.var_flat))

    for n in eachindex(mapping.shift.st)

        d_max = Bs[n].op_range.d_max
        κ_λ_lb = Bs[n].op_range.κ_λ_lb
        κ_λ_ub = Bs[n].op_range.κ_λ_ub

        # shift.
        for i in eachindex(mapping.shift.st[n])

            #r_lb = 2*π*(u_min - d_max[i])
            #r_ub = 2*π*(u_max + d_max[i])
            ζ_max = d_max[i]*twopi(T)*shift_proportion

            for l = mapping.shift.st[n][i]:mapping.shift.fin[n][i]
                lbs[l] = -ζ_max #r_lb
                ubs[l] = ζ_max #r_ub
            end
        end

        # phase.
        for i in eachindex(mapping.phase.st[n])

            for l = mapping.phase.st[n][i]:mapping.phase.fin[n][i]
                lbs[l] = phase_lb
                ubs[l] = phase_ub
            end
        end

        # T2 multiplier.
        for i in eachindex(mapping.T2.st[n])

            for l = mapping.T2.st[n][i]:mapping.T2.fin[n][i]
                lbs[l] = κ_λ_lb
                ubs[l] = κ_λ_ub
            end
        end
    end

    return lbs, ubs
end

function initializeparameter(αs::Vector{Vector{T}}, default_val::T) where T <: AbstractFloat
    out = similar(αs)
    
    for n in eachindex(αs)
        out[n] = Vector{T}(undef, length(αs[n]))
        fill!(out[n], default_val)
    end

    return out
end



# """
#     gettimerange(N::Int, fs::T) where T

# Returns the time stamps for a sequence, starting at time 0. Returns zero(T):Ts:(N-1)*Ts, Ts = 1/fs.
# """
function gettimerange(N::Int, fs::T) where T
    Ts::T = 1/fs

    return zero(T):Ts:(N-1)*Ts
end

# """
#     getDFTfreqrange(N::Int, fs::T) where T

# Returns the frequency stamps for a DFT sequence, computed by fft().
#     Starting at frequency 0 Hz. Returns LinRange(0, fs-fs/N, N).
# """
function getDFTfreqrange(N::Int, fs::T)::LinRange{T} where T
    a = zero(T)
    b = fs-fs/N

    return LinRange(a, b, N)
end


function verifyupdatedvars(
    p,
    phases::Vector{PT};
    st_ind = 1,
    ) where PT <: CoherenceParams

    discrepancy = 0.0
    l = st_ind

    for n in eachindex(phases)
        for i in eachindex(phases[n].var)
            for j in eachindex(phases[n].var[i])
                
                discrepancy += abs(phases[n].var[i][j] - p[begin+l-1])
                l += 1
            end
        end
    end

    return discrepancy, l-1
end

function verifyupdatedvars(
    p,
    T2s::Vector{ST};
    st_ind = 1,
    ) where ST <: SharedParams

    discrepancy = 0.0
    l = st_ind 

    for n in eachindex(T2s)
        for i in eachindex(T2s[n].var)
            for j in eachindex(T2s[n].var[i])
                
                discrepancy += abs(T2s[n].var[i][j] - p[begin+l-1])
                l += 1
            end
        end
    end

    return discrepancy, l-1
end


#### testing

function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generateparameters(lbs::Vector{T}, ubs::Vector{T})::Vector{T} where T
    
    @assert length(lbs) == length(ubs)

    return collect( convertcompactdomain(rand(T), zero(T), one(T), lbs[i], ubs[i]) for i in eachindex(lbs) )
end
