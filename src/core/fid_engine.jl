# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# # Free-induction decay (FID) surrogate (time-domain).

@kwdef struct FIDSurrogateConfig{T}
    Δt::T = convert(T, 1e-5)
    t_lb::T = zero(T)
    t_ub::T = convert(T, 3.0)
end


function create_fid_surrogate(As::Vector{NMRHamiltonian.SHType{T}}, λ0::T, config::FIDSurrogateConfig{T}) where T <: AbstractFloat #where {T <: AbstractFloat, ST,PT,T2T}

    qs = compute_rg_fid_surrogates(As, config)

    model = FIDSurrogateModel(qs, ResonanceGroupDCs(As), λ0, FrequencyConversion(As[begin]))
    rcs_nested, _ = create_rcs_list(As)
    info = FIDSurrogateInfo(config.t_lb, config.t_ub, get_rg_flat_mapping(As), rcs_nested)
    return model, info
end


function get_fid_itplocations(C::FIDSurrogateConfig{T}) where T <: AbstractFloat

    # Since we're using bi-cubic B-splines for interpolation, we want to avoid border artefacts in our interested range.
    # The border for bi-cubic B-splines interpolation for grid-sampled data is 4 samples.
    t_border = C.Δt*4 + eps(T)*100
    # r_border = 0
    # κ_λ_border = 0

    # LinRange is type stable
    return step2LinRange(C.t_lb - t_border, C.Δt, C.t_ub + t_border)
end

function compute_rg_fid_surrogates(As::Vector{NMRHamiltonian.SHType{T}}, config::FIDSurrogateConfig{T}) where T <: AbstractFloat

    A_t = get_fid_itplocations(config)

    qs = Memory{Memory{Memory{ITP.Interpolator1DComplex{T}}}}(undef, length(As))

    for n in eachindex(As)
        qs[n] = Memory{Memory{ITP.Interpolator1DComplex{T}}}(undef, length(As[n].Δc_bar))

        for i in eachindex(As[n].Δc_bar)
            qs[n][i] = Memory{ITP.Interpolator1DComplex{T}}(undef, length(As[n].Δc_bar[i]))

            for k in eachindex(As[n].Δc_bar[i])
                
                inds = As[n].parts[i][k]
                α = As[n].αs[i][inds]
                Ω = As[n].Ωs[i][inds]
                samples = compute_fid_samples(α, Ω, A_t)

                qs[n][i][k] = compute_1D_itp(samples, A_t)
            end
        end
    end

    return qs
end

function compute_fid_samples(α::Vector{T}, Ω::Vector{T}, A_t) where T <: AbstractFloat
    return [ evalfidpart(t, α, Ω) for t in A_t ] # complex-valued.
end

function evalfidpart(t::T, α::Vector{T}, Ω::Vector{T}) where T <: AbstractFloat
    return sum( α[l]*cis(Ω[l]*t) for l in eachindex(α,Ω) )
end

function compute_1D_itp(A::Vector{Complex{T}}, A_t) where T <: AbstractFloat
    Sr = real.(A)
    Si = imag.(A)
    cbuf = ITP.FitBuffer1D(T, length(Sr); N_padding = 10)
    return ITP.Interpolator1DComplex(ITP.LinearPadding(), ITP.ConstantExtrapolation(), cbuf, Sr, Si, first(A_t), last(A_t); ϵ = eps(T)*100)
end

