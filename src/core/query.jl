# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>


### prelim model.

function eval_cl(
    angular_freq,
    ws::AbstractVector{T},
    As::AbstractVector{HT},
    s::SystemT2Cache{T},
    ) where {T <: AbstractFloat, HT <: NMRHamiltonian.SHType{T}}

    (length(ws) == length(As) == length(s.rg) == length(s.λs)) || error("Length mismatch")
    rg, λs = s.rg, s.λs

    out = zero(Complex{T})
    for n in eachindex(As, rg, λs, ws)
        rg_n, λ_n = rg[n], λs[n]
        A = As[n]

        out_n = zero(Complex{T})
        for i in eachindex(A.parts, A.αs, A.Ωs, rg_n, λ_n)
            rg_ni = rg_n[i]
            λ_ni = λ_n[i]
            parts_i, α_i, Ω_i = A.parts[i], A.αs[i], A.Ωs[i]

            for k in eachindex(parts_i, rg_ni)
                # parse
                tmp = rg_ni[k]
                #ζ, cos_β, sin_β = tmp.ζ, tmp.cos_β, tmp.sin_β
                ζ, cis_β = tmp.ζ, tmp.cis_β
                inds = parts_i[k]

                out_n += evalclpart(angular_freq - ζ, α_i[inds], Ω_i[inds], λ_ni)*cis_β
            end
        end

        out += ws[n]*out_n
    end

    return out
end

function eval_cl_rg(
    angular_freq,
    ws::AbstractVector{T},
    As::AbstractVector{HT},
    s::SystemT2Cache{T},
    compound_index::Integer,
    spin_sys_index::Integer,
    group_index::Integer,
    ) where {T <: AbstractFloat, HT <: NMRHamiltonian.SHType{T}}

    (length(ws) == length(As) == length(s.rg) == length(s.λs)) || error("Length mismatch")
    rg, λs = s.rg, s.λs

    out = zero(Complex{T})
    n = compound_index
    i = spin_sys_index
    k = group_index

    # # n
    rg_n, λ_n = rg[n], λs[n]
    A = As[n]

    out_n = zero(Complex{T})
    
    # # i
    rg_ni = rg_n[i]
    λ_ni = λ_n[i]
    parts_i, α_i, Ω_i = A.parts[i], A.αs[i], A.Ωs[i]

    # # k
    # parse
    tmp = rg_ni[k]
    #ζ, cos_β, sin_β = tmp.ζ, tmp.cos_β, tmp.sin_β
    ζ, cis_β = tmp.ζ, tmp.cis_β
    inds = parts_i[k]

    out_n += evalclpart(angular_freq - ζ, α_i[inds], Ω_i[inds], λ_ni)*cis_β

    # # n
    out += ws[n]*out_n

    return out
end

### surrogate model

# for a nested qs.
function eval_cl(
    angular_freq::T,
    ws::AbstractVector{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    s::SystemT2Cache{T},
    ) where T <: AbstractFloat

    (length(ws) == length(qs) == length(s.rg) == length(s.λs)) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    out = zero(Complex{T})
    for n in eachindex(qs, rg, λs, ws)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]

        out_n = zero(Complex{T})
        for i in eachindex(qs_n, rg_n, λ_n)
            qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
            
            for k in eachindex(qs_ni, rg_ni)

                # tmp = rg_ni[k]
                # #ζ, cos_β, sin_β = tmp.ζ, tmp.cos_β, tmp.sin_β
                # ζ, cis_β = tmp.ζ, tmp.cis_β

                # out_n += eval_surrogate_cl(qs_ni[k], angular_freq - ζ, λ_ni, cis_β)
                out_n += eval_surrogate_cl(qs_ni[k], angular_freq, λ_ni, rg_ni[k])
                #out_n += eval_surrogate_cl(qs_ni[k], angular_freq - rg_ni[k].ζ, λ_ni, rg_ni[k].cis_β)
            end
        end

        out += ws[n]*out_n
    end

    return out
end


function eval_surrogate_cl(A::ITP.Interpolator2DComplex{T}, u_rad::T, λ::T, B::RGState{T}) where T <: AbstractFloat
    return ITP.query2D(u_rad - B.ζ, λ, A)*B.cis_β
end

# used for gradient.
function eval_surrogate_cl_separate(A::ITP.Interpolator2DComplex{T}, u_rad::T, λ::T, B::RGState{T}) where T <: AbstractFloat
    itp_eval = ITP.query2D(u_rad - B.ζ, λ, A)
    surrogate_out = itp_eval*B.cis_β
    return surrogate_out, itp_eval
end

# batch

# Compute the design matrix (stored in `out`) given `U_rad`. One column per entry in `qs`.
function eval_cl_matrix!(
    out::Matrix{Complex{T}}, # mutates
    U_rad::Memory{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    s::SystemT2Cache{T},
    ) where T <: AbstractFloat

    size(out,1) == length(U_rad) || error("Input and output buffer length mismatch.")
    size(out,2) == length(qs) == length(s.rg) == length(s.λs) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    fill!(out, zero(Complex{T}))
    for n in eachindex(qs, rg, λs)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]

        for i in eachindex(qs_n, rg_n, λ_n)
            qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
            
            for k in eachindex(qs_ni, rg_ni)
                for m in eachindex(U_rad)
                    out[m, n] += eval_surrogate_cl(qs_ni[k], U_rad[m], λ_ni, rg_ni[k])
                end
            end
        end
    end

    return nothing
end

function eval_cl_batch!(
    out::Memory{Complex{T}}, # mutates
    buffer::Memory{Complex{T}}, # mutates
    U_rad::Memory{T},
    ws::AbstractVector{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    s::SystemT2Cache{T},
    ) where T <: AbstractFloat

    length(out) == length(buffer) == length(U_rad) || error("Input and output buffer length mismatch.")
    length(ws) == length(qs) == length(s.rg) == length(s.λs) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    fill!(out, zero(Complex{T}))
    for n in eachindex(qs, rg, λs, ws)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]

        fill!(buffer, zero(Complex{T}))
        for i in eachindex(qs_n, rg_n, λ_n)
            qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
            
            for k in eachindex(qs_ni, rg_ni)
                for m in eachindex(U_rad, buffer)
                    buffer[m] += eval_surrogate_cl(qs_ni[k], U_rad[m], λ_ni, rg_ni[k])
                end
            end
        end

        out .+= ws[n] .* buffer
    end

    return nothing
end

# slower, no buffer memory requirement.
function eval_cl_batch!(
    out::Memory{Complex{T}}, # mutates
    U_rad::Memory{T},
    ws::AbstractVector{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    state::SystemT2Cache{T},
    ) where T <: AbstractFloat

    for m in eachindex(out, U_rad)
        out[m] = eval_cl(U_rad[m], ws, qs, state)
    end

    return nothing
end


############## FID

### prelim model.

function eval_fid(
    t::T,
    ws::AbstractVector{T},
    As::AbstractVector{HT},
    s::SystemT2Cache{T},
    ) where {T <: AbstractFloat, HT <: NMRHamiltonian.SHType{T}}

    (length(ws) == length(As) == length(s.rg) == length(s.λs)) || error("Length mismatch")
    rg, λs = s.rg, s.λs

    out = zero(Complex{T})
    for n in eachindex(As, rg, λs, ws)
        rg_n, λ_n = rg[n], λs[n]
        A = As[n]

        out_n = zero(Complex{T})
        for i in eachindex(A.parts, A.αs, A.Ωs, rg_n, λ_n)
            rg_ni = rg_n[i]
            parts_i, α_i, Ω_i = A.parts[i], A.αs[i], A.Ωs[i]

            out_ni = zero(Complex{T})
            for k in eachindex(parts_i, rg_ni)
                # parse
                tmp = rg_ni[k]
                #ζ, cos_β, sin_β = tmp.ζ, tmp.cos_β, tmp.sin_β
                ζ, cis_β = tmp.ζ, tmp.cis_β
                inds = parts_i[k]

                out_ni += evalfidpart(t, α_i[inds], Ω_i[inds])*cis_β*cis(ζ*t)
            end

            out_n += out_ni*exp(-λ_n[i]*t)
        end

        out += ws[n]*out_n
    end

    return out
end

### surrogate model


# for a nested qs.
function eval_fid(
    t::T,
    ws::AbstractVector{T},
    qs::Memory{Memory{Memory{ITP.Interpolator1DComplex{T}}}},
    s::SystemT2Cache{T},
    ) where T <: AbstractFloat

    (length(ws) == length(qs) == length(s.rg) == length(s.λs)) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    out = zero(Complex{T})
    for n in eachindex(qs, rg, λs, ws)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]

        out_n = zero(Complex{T})
        for i in eachindex(qs_n, rg_n, λ_n)
            qs_ni, rg_ni = qs_n[i], rg_n[i]
            
            out_ni = zero(Complex{T})
            for k in eachindex(qs_ni, rg_ni)

                # tmp = rg_ni[k]
                # #ζ, cos_β, sin_β = tmp.ζ, tmp.cos_β, tmp.sin_β
                # ζ, cis_β = tmp.ζ, tmp.cis_β

                # out_n += eval_surrogate_cl(qs_ni[k], angular_freq - ζ, λ_ni, cis_β)
                out_ni += eval_surrogate_fid(qs_ni[k], t, rg_ni[k])
                #out_n += eval_surrogate_fid(qs_ni[k], angular_freq - rg_ni[k].ζ, λ_ni, rg_ni[k].cis_β)
            end

            out_n += out_ni*exp(-λ_n[i]*t)
        end

        out += ws[n]*out_n
    end

    return out
end

# doesn't include eps(-λ*t).
function eval_surrogate_fid(A::ITP.Interpolator1DComplex{T}, t::T, B::RGState{T}) where T <: AbstractFloat
    return ITP.query1D(t, A)*B.cis_β*cis(B.ζ*t)
end

#### Front end and derivatives

struct QueryState{ST <: SurrogateCache, PT <: SurrogateParameters, MT <: SurrogateModel}
    system_cache::ST
    p_cache::PT
    model::MT
end

function get_prelim_state(A::QueryState)
    return A.system_cache
end

# creates a reference to `model`. It will mutate `model` if the output `QueryState` is mutated.
function create_state_reference(model::MT, model_info::SurrogateInfo) where MT <: SurrogateModel
    s = SystemT2Cache(model, model_info)
    p, _ = initialize_params(SystemT2(), model_info, model)
    return QueryState(s, p, model)
end

function update_state!(s::QueryState, v::AbstractVector)
    update_parameters!(s.p_cache, v)
    update_state!(s.system_cache, s.p_cache, s.model)
    return nothing
end

function update_state!(s::QueryState, x::SurrogateParameters)
    return update_state!(s, x.contents)
end

# mutates `state` and `state`.
function eval_cl!(s::QueryState, u_rad::T, v::AbstractVector{T}) where T <: AbstractFloat
    update_state!(s, v)
    return eval_cl(s, u_rad)
end

function eval_cl(s::QueryState, u_rad::AbstractFloat)
    return eval_cl(u_rad, get_concentration(s.p_cache), get_proxies(s.model), s.system_cache)
end

function eval_fid(s::QueryState, t::AbstractFloat)
    return eval_fid(t, get_concentration(s.p_cache), get_proxies(s.model), s.system_cache)
end

function eval_cl_rg(
    angular_freq,
    ws::AbstractVector{T},
    As::AbstractVector{HT},
    s::QueryState,
    compound_index::Integer,
    spin_sys_index::Integer,
    group_index::Integer,
    ) where {T <: AbstractFloat, HT <: NMRHamiltonian.SHType{T}}

    return eval_cl_rg(angular_freq, ws, As, s.system_cache, compound_index, spin_sys_index, group_index)
end