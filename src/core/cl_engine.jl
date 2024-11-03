# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# # Complex lorentzian (CL) surrogates (frequency-domain).

"""
    struct CLSurrogateConfig{T}
        Δr::T = convert(T, 1.0)
        Δκ_λ::T = convert(T, 0.05)
        κ_λ_lb::T = convert(T, 0.5)
        κ_λ_ub::T = convert(T, 2.5)
        Δppm_border::T = convert(T, 0.5)
    end

- `κ_λ_lb`, `κ_λ_ub` -- the default lower and upper bounds, respectively, for the κ_λ input of the surrogate.

- `Δppm_border` controls the perturbation from the reference resonance frequencies. Measured in ppm.

- `Δr`, `Δκ_λ` -- the sampling increment for the frequency input r and T2 multiplier input κ_λ for generating samples to fit the surrogate. Smaller means the surrogate is more accurate, but slower to construct the surrogate.
"""
@kwdef struct CLSurrogateConfig{T}
    Δr::T = convert(T, 1.0) # the samples used to build the surrogate is taken every `Δr` radian on the frequency axis. Decrease for improved accuracy at the expense of computation resources.
    Δκ_λ::T = convert(T, 0.05) # the samples used to build thes urrogate for κ_λ are taken at this sampling spacing. Decrease for improved accuracy at the expense of computation resources.
    
    κ_λ_lb::T = convert(T, 0.5) # interpolation lower limit for κ_λ.
    κ_λ_ub::T = convert(T, 2.5) # interpolation upper limit for κ_λ.

    Δppm_border::T = convert(T , 0.5)
end

"""
    create_cl_surrogate(
        As::Vector{NMRHamiltonian.SHType{T}},
        λ0::T,
        config::CLSurrogateConfig{T};
        names::Vector{String} = Vector{String}(undef, 0),
    ) where T <: AbstractFloat

Create surrogate for NMR spectrum, given the simulated resonance comopnent results from NMRHamiltonian.

Inputs:

- `As` -- a 1-D array of compound resonance simulations. See `NMRHamiltonian.simulate`.

- `λ0` -- the estimated T2* decay for the 0 ppm resonance component.

- `config` -- the configuration for building the surrogate. See `CLSurrogateConfig`.
"""
function create_cl_surrogate(As::Vector{NMRHamiltonian.SHType{T}}, λ0::T, config::CLSurrogateConfig{T}) where T <: AbstractFloat #where {T <: AbstractFloat, ST,PT,T2T}

    # get global frequency interval, over which the surrogate must be valid.
    hz_lb, hz_ub = get_hz_bounds(As, config.Δppm_border)
    
    qs = compute_rg_cl_surrogates(As, λ0, config, hz_lb, hz_ub)

    model = CLSurrogateModel(qs, ResonanceGroupDCs(As), λ0, FrequencyConversion(As[begin]))
    
    rcs_nested, _ = create_rcs_list(As)
    info = CLSurrogateInfo(hz_lb, hz_ub, get_rg_flat_mapping(As), rcs_nested)
    return model, info
end

function compute_2D_itp(A::Matrix{Complex{T}}, A_r, A_λ) where T <: AbstractFloat
    Sr = real.(A)
    Si = imag.(A)
    cbuf = ITP.FitBuffer2D(T, size(Sr); N_padding = (10,10))

    return ITP.Interpolator2DComplex(
        ITP.LinearPadding(),
        ITP.ConstantExtrapolation(),
        cbuf, Sr, Si,
        first(A_r), last(A_r), first(A_λ), last(A_λ);
        ϵ = eps(T)*100,
    )
end

function compute_rg_cl_surrogates(
    As::Vector{NMRHamiltonian.SHType{T}},
    λ0::T,
    config::CLSurrogateConfig{T},
    hz_lb::T,
    hz_ub::T
    ) where T <: AbstractFloat

    A_r, A_λ = get_cl_itplocations(hz_lb, hz_ub, λ0, config)

    qs = Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}}(undef, length(As))

    for n in eachindex(As)
        qs[n] = Memory{Memory{ITP.Interpolator2DComplex{T}}}(undef, length(As[n].Δc_bar))

        for i in eachindex(As[n].Δc_bar)
            qs[n][i] = Memory{ITP.Interpolator2DComplex{T}}(undef, length(As[n].Δc_bar[i]))

            for k in eachindex(As[n].Δc_bar[i])
                
                inds = As[n].parts[i][k]
                α = As[n].αs[i][inds]
                Ω = As[n].Ωs[i][inds]
                samples = compute_cl_samples(α, Ω, A_r, A_λ)

                qs[n][i][k] = compute_2D_itp(samples, A_r, A_λ)
            end
        end
    end

    return qs
end

function get_hz_bounds(As::AbstractVector{AT}, Δppm_border::T) where {T <: AbstractFloat, AT <: NMRHamiltonian.SHType}
    
    @assert !isempty(As)
    fc = FrequencyConversion(As[begin])
    hz2ppmfunc = uu->hz_to_ppm(uu, fc)

    min_Ω = convert(T, Inf)
    max_Ω = convert(T, -Inf)

    for A in As
        for i in eachindex(A.Ωs)
            for l in eachindex(A.Ωs[i])
                min_Ω = min(min_Ω, A.Ωs[i][l])
                max_Ω = max(max_Ω, A.Ωs[i][l])
            end
        end
    end
    min_ppm = hz2ppmfunc(min_Ω/twopi(T)) - Δppm_border
    max_ppm = hz2ppmfunc(max_Ω/twopi(T)) + Δppm_border

    return ppm_to_hz(min_ppm, fc), ppm_to_hz(max_ppm, fc)
end

# Don't pre-allocate A since we'll do multi-threaded later at the call site.
function compute_cl_samples(α::Vector{T}, Ω::Vector{T}, A_r, A_λ) where T <: AbstractFloat
    return [ evalclpart(r, α, Ω, λ) for r in A_r, λ in A_λ] # complex-valued.
end

function evalclpart(r::T, α::Vector{T}, Ω::Vector{T}, λ::T) where T <: AbstractFloat
    return sum( α[l]/(λ+im*(r-Ω[l])) for l in eachindex(α,Ω) )
end


# for use with CubicBSplineInterpolation.jl
function get_cl_itplocations(hz_lb::T, hz_ub::T, λ0::T, C::CLSurrogateConfig{T}) where T <: AbstractFloat
    
    κ_λ_lb, κ_λ_ub, Δr, Δκ_λ = C.κ_λ_lb, C.κ_λ_ub, C.Δr, C.Δκ_λ

    r_min = twopi(T)*hz_lb
    r_max = twopi(T)*hz_ub

    # Since we're using bi-cubic B-splines for interpolation, we want to avoid border artefacts in our interested range.
    # The border for bi-cubic B-splines interpolation for grid-sampled data is 4 samples.
    r_border = Δr*4 + eps(T)*100
    κ_λ_border = Δκ_λ*4 + eps(T)*100
    # r_border = 0
    # κ_λ_border = 0

    # LinRange is type stable
    A_r = step2LinRange(r_min - r_border, Δr, r_max + r_border) # large range.
    A_λ = step2LinRange(
        (κ_λ_lb - κ_λ_border)*λ0,
        Δκ_λ*λ0,
        (κ_λ_ub + κ_λ_border)*λ0,
    )

    return A_r, A_λ
end

function step2LinRange(x1::T, s::T, x2::T) where T <: AbstractFloat
    N_steps = floor(Int, (x2+s-x1)/s)
    y1 = x1
    y2 = convert(T, x1 + s*(N_steps-1))
    return LinRange(y1, y2, N_steps)
end
# x1, s, x2 = (80692.28f0, 1.0f0, 87574.445f0)
# r = x1:s:x2
# N_steps = floor(Int, (x2+s-x1)/s)
# y1 = x1
# y2 = convert(T, x1 + s*(N_steps-1))
# q = LinRange(y1,y2,N_steps)
# @assert norm(collect(r) - collect(q)) < 1e-8@show


