# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

function eval_cl_dq_nik(A::ITP.Interpolator2DComplex{T}, u_rad::T, λ::T, B::RGState{T}) where T <: AbstractFloat
    return ITP.query2D_derivative1(u_rad - B.ζ, λ, A)
end

function eval_cl_and_gradient!(
    gr::SurrogateParameters,
    gi::SurrogateParameters,
    s::QueryState,
    u_rad::AbstractFloat,
    v::AbstractVector)

    update_state!(s, v)
    return eval_cl_and_gradient!(gr, gi, s, u_rad)
end

function eval_cl_and_gradient!(
    gr::SurrogateParameters, # mutates, overwrites, output.
    gi::SurrogateParameters, # mutates, overwrites, output.
    ps::QueryState, # doesn't mutate. input,
    u_rad::T,
    ) where T <: AbstractFloat

    s = ps.system_cache
    ws = get_concentration(ps.p_cache)

    model = ps.model
    qs = get_proxies(model)
    cbs = get_Δc(model)
    cbr = get_nested(cbs)
    λ0 = get_λ0(model)

    length(qs) == length(s.rg) == length(s.λs) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    l = 0 # spin system counter.

    # reset
    out = zero(Complex{T}) # evaluation of surrogate.
    fill!(gr.contents, zero(T))
    fill!(gi.contents, zero(T))

    # update gradient, eval surrogate.
    for n in eachindex(qs, rg, λs, ws, cbr)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]
        cbr_n = cbr[n]

        out_n = zero(Complex{T})

        for i in eachindex(qs_n, rg_n, λ_n, cbr_n)
            qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
            Δc_bar_mat = cbr_n[i]

            # shift, phase are vectors in a spin system.
            l += 1
            ∂ψr_∂z = gr.shift[l] # denote shift parameters, z.
            ∂ψr_∂b = gr.phase[l] # denote phase parameters, b.
            ∂ψi_∂z = gi.shift[l] # denote shift parameters, z.
            ∂ψi_∂b = gi.phase[l] # denote phase parameters, b.

            # For a spin system, decay is a scalar.
            ∂ψr_∂κ = zero(T) # denote decay parameters, κ.
            ∂ψi_∂κ = zero(T) # denote decay parameters, κ.

            for k in eachindex(qs_ni, rg_ni)

                Δc_bar = view(Δc_bar_mat, k, :)

                # unpack
                cos_β = real(rg_ni[k].cis_β)
                sin_β = imag(rg_ni[k].cis_β)

                # get itp evals.
                q_eval, itp_eval = eval_surrogate_cl_separate(qs_ni[k], u_rad, λ_ni, rg_ni[k])
                qr = real(itp_eval)
                qi = imag(itp_eval)
                ∂qr_∂r, ∂qr_∂λ, ∂qi_∂r, ∂qi_∂λ = eval_cl_dq_nik(qs_ni[k], u_rad, λ_ni, rg_ni[k])

                # ## chain rule
                # shifts
                for d in eachindex(∂ψr_∂z, ∂ψi_∂z, Δc_bar)

                    ∂qr_∂z_d = -Δc_bar[d] * ∂qr_∂r
                    ∂qi_∂z_d = -Δc_bar[d] * ∂qi_∂r

                    ∂ψr_∂z[d] += ∂qr_∂z_d * cos_β - ∂qi_∂z_d * sin_β
                    ∂ψi_∂z[d] += ∂qr_∂z_d * sin_β + ∂qi_∂z_d * cos_β
                end

                # phase
                for d in eachindex(∂ψr_∂b, ∂ψi_∂b, Δc_bar)

                    ∂β_∂b_d = Δc_bar[d]

                    ∂ψr_∂b[d] += (-qr * sin_β - qi * cos_β)*∂β_∂b_d
                    ∂ψi_∂b[d] += (qr * cos_β - qi * sin_β)*∂β_∂b_d
                end

                # decay
                ∂λ_∂κ = λ0
                ∂qr_∂κ = ∂λ_∂κ* ∂qr_∂λ
                ∂qi_∂κ = ∂λ_∂κ* ∂qi_∂λ

                ∂ψr_∂κ += ∂qr_∂κ * cos_β - ∂qi_∂κ * sin_β
                ∂ψi_∂κ += ∂qr_∂κ * sin_β + ∂qi_∂κ * cos_β

                # eval surrogate
                out_n += q_eval
            end

            # decay
            gr.decay_multiplier[l] = ∂ψr_∂κ *ws[n]
            gi.decay_multiplier[l] = ∂ψi_∂κ *ws[n]

            # adjust for concentration.
            ∂ψr_∂z .= ∂ψr_∂z .* ws[n] 
            ∂ψr_∂b .= ∂ψr_∂b .* ws[n] 
            ∂ψi_∂z .= ∂ψi_∂z .* ws[n] 
            ∂ψi_∂b .= ∂ψi_∂b .* ws[n]
        end

        # gradient
        gr.concentration[n] = real(out_n)
        gi.concentration[n] = imag(out_n)

        # eval surrogate
        out += ws[n]*out_n
    end

    return out
end

# based on eval_cl_matrix!
# Compute the design matrix (stored in `out`) given `U_rad`. One column per entry in `qs`.
function eval_cl_matrices!(
    out::Matrix{Complex{T}}, # mutates, output
    #q_evals::Matrix{Complex{T}}, # mutates, output
    itp_evals::Matrix{Complex{T}}, # mutates, output
    U_rad::Memory{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    s::SystemT2Cache{T},
    ) where T <: AbstractFloat

    size(itp_evals, 1) == size(out,1) || error("Size mistmatch.")
    size(out,1) == length(U_rad) == size(itp_evals,1) || error("Input and output buffer length mismatch.")
    size(out,2) == length(qs) == length(s.rg) == length(s.λs) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    l = 0

    fill!(out, zero(Complex{T}))
    for n in eachindex(qs, rg, λs)
        rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]

        for i in eachindex(qs_n, rg_n, λ_n)
            qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
            
            for k in eachindex(qs_ni, rg_ni)

                l += 1
                for m in eachindex(U_rad)
                    q_eval, itp_eval = eval_surrogate_cl_separate(qs_ni[k], U_rad[m], λ_ni, rg_ni[k])

                    out[m, n] += q_eval
                    #q_evals[m,l] = q_eval
                    itp_evals[m,l] = itp_eval
                end
            end
        end
    end

    return nothing
end

# based on eval_cl_and_gradient! and 
function eval_bls_gradient!(
    df_dp::SurrogateParameters, # mutates, output
    gr::SurrogateParameters, # mutates, buffer.
    gi::SurrogateParameters, # mutates, buffer.
    s::SystemT2Cache, # not mutating. input
    itp_evals::Matrix{Complex{T}},
    U_rad::Union{LinRange, Memory},
    ws::Memory{T},
    r_interlaced::Memory{T},
    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}},
    cbs::ResonanceGroupDCs,
    λ0::T,
    ) where T <: AbstractFloat

    length(get_resonance_contents(df_dp)) == length(get_resonance_contents(gi)) == length(get_resonance_contents(gr)) || error("Length mismatch.")

    r = reinterpret(Complex{T}, r_interlaced)
    length(r) == length(U_rad) == size(itp_evals, 1) || error("Size mismatch.") # M
    
    ## 
    #s = ps.system_cache

    #qs = get_proxies(model)
    #cbs = get_Δc(model)
    cbr = get_nested(cbs)
    #λ0 = get_λ0(model)

    length(qs) == length(s.rg) == length(s.λs) || error("State buffer length mismatch")
    rg, λs = s.rg, s.λs

    df_dp_rc = get_resonance_contents(df_dp)
    gr_rc = get_resonance_contents(gr)
    gi_rc = get_resonance_contents(gi)
    
    fill!(df_dp_rc, zero(T))
    for m in eachindex(U_rad, r)
        u_rad = U_rad[m]

        fill!(gr_rc, zero(T))
        fill!(gi_rc, zero(T))
        l = 0 # spin system counter.
        l2 = 0 # resonance group counter
        #out = zero(Complex{T}) # evaluation of surrogate.

        # update gradient. Based on eval_cl_and_gradient!
        for n in eachindex(qs, rg, λs, ws, cbr)
            rg_n, λ_n, qs_n = rg[n], λs[n], qs[n]
            cbr_n = cbr[n]

            #out_n = zero(Complex{T})

            for i in eachindex(qs_n, rg_n, λ_n, cbr_n)
                qs_ni, rg_ni, λ_ni = qs_n[i], rg_n[i], λ_n[i]
                Δc_bar_mat = cbr_n[i]

                # shift, phase are vectors in a spin system.
                l += 1
                ∂ψr_∂z = gr.shift[l] # denote shift parameters, z.
                ∂ψr_∂b = gr.phase[l] # denote phase parameters, b.
                ∂ψi_∂z = gi.shift[l] # denote shift parameters, z.
                ∂ψi_∂b = gi.phase[l] # denote phase parameters, b.

                # For a spin system, decay is a scalar.
                ∂ψr_∂κ = zero(T) # denote decay parameters, κ.
                ∂ψi_∂κ = zero(T) # denote decay parameters, κ.

                for k in eachindex(qs_ni, rg_ni)

                    Δc_bar = view(Δc_bar_mat, k, :)

                    # unpack
                    cos_β = real(rg_ni[k].cis_β)
                    sin_β = imag(rg_ni[k].cis_β)

                    # get itp evals.
                    # q_eval, itp_eval = eval_surrogate_cl_separate(
                    #     qs_ni[k], u_rad, λ_ni, rg_ni[k],
                    # )
                    # qr = real(itp_eval)
                    # qi = imag(itp_eval)
                    l2 += 1
                    qr = real(itp_evals[m,l2]) # TODO improve this inefficient memory access
                    qi = imag(itp_evals[m,l2]) # TODO improve this inefficient memory access
                    
                    ∂qr_∂r, ∂qr_∂λ, ∂qi_∂r, ∂qi_∂λ = eval_cl_dq_nik(qs_ni[k], u_rad, λ_ni, rg_ni[k])

                    # ## chain rule
                    # shifts
                    for d in eachindex(∂ψr_∂z, ∂ψi_∂z, Δc_bar)

                        ∂qr_∂z_d = -Δc_bar[d] * ∂qr_∂r
                        ∂qi_∂z_d = -Δc_bar[d] * ∂qi_∂r

                        ∂ψr_∂z[d] += ∂qr_∂z_d * cos_β - ∂qi_∂z_d * sin_β
                        ∂ψi_∂z[d] += ∂qr_∂z_d * sin_β + ∂qi_∂z_d * cos_β
                    end

                    # phase
                    for d in eachindex(∂ψr_∂b, ∂ψi_∂b, Δc_bar)

                        ∂β_∂b_d = Δc_bar[d]

                        ∂ψr_∂b[d] += (-qr * sin_β - qi * cos_β)*∂β_∂b_d
                        ∂ψi_∂b[d] += (qr * cos_β - qi * sin_β)*∂β_∂b_d
                    end

                    # decay
                    ∂λ_∂κ = λ0
                    ∂qr_∂κ = ∂λ_∂κ* ∂qr_∂λ
                    ∂qi_∂κ = ∂λ_∂κ* ∂qi_∂λ

                    ∂ψr_∂κ += ∂qr_∂κ * cos_β - ∂qi_∂κ * sin_β
                    ∂ψi_∂κ += ∂qr_∂κ * sin_β + ∂qi_∂κ * cos_β

                    # # # eval surrogate
                    # out_n += q_eval
                end

                # decay
                gr.decay_multiplier[l] = ∂ψr_∂κ *ws[n]
                gi.decay_multiplier[l] = ∂ψi_∂κ *ws[n]

                # adjust for concentration.
                ∂ψr_∂z .= ∂ψr_∂z .* ws[n] 
                ∂ψr_∂b .= ∂ψr_∂b .* ws[n] 
                ∂ψi_∂z .= ∂ψi_∂z .* ws[n] 
                ∂ψi_∂b .= ∂ψi_∂b .* ws[n]
            end

            # # gradient
            # gr.concentration[n] = real(out_n)
            # gi.concentration[n] = imag(out_n)

            # # eval surrogate
            # out += ws[n]*out_n
        end

        df_dp_rc .+= real(r[m]) .* gr_rc .+ imag(r[m]) .* gi_rc
    end

    df_dp_rc .= 2 .* df_dp_rc

    return nothing
    #return out
end