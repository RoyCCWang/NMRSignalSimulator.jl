# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

#### least squares, no concentration parameter.
abstract type AbstractNLSCostParameters end

struct NLSCostParameters{T <: AbstractFloat, PT, FT} <: AbstractNLSCostParameters
    #qc::QuadraticCostParameters{T}

    U_rad::Memory{T}
    #y::Memory{Complex{T}}
    yr::Memory{T}

    qs::Memory{Memory{Memory{ITP.Interpolator2DComplex{T}}}}
    Δc_bars::ResonanceGroupDCs{T}
    λ0::T

    nls_callable!::FT # this should not mutate when calling NLSCostCallable.
    nls_params::PT

    function NLSCostParameters(
        U_hz::AbstractVector{T},
        y::AbstractVector{Complex{T}},
        model::CLSurrogateModel,
        nls_params::PT,
        nls_callable!::FT) where {T <: AbstractFloat, PT, FT}
        
        length(U_hz) == length(y) || error("Length mismatch")
        U_rad = Memory{T}(undef, length(U_hz))
        U_rad .= T(2)*T(π) .* U_hz

        return new{T,PT, FT}(
            U_rad,
            #Memory{Complex{T}}(y),
            Memory{T}(reinterpret(T, y)),
            get_proxies(model),
            get_Δc(model),
            get_λ0(model),
            nls_callable!,
            nls_params,
        )
    end
end

# mutates when evaluating QuadraticCostCallable.
abstract type AbstractNLSCostState end

struct NLSCostState{T <: AbstractFloat, PT <: SurrogateParameters, NT} <: AbstractNLSCostState

    design_matrix::Matrix{Complex{T}}
    model_state::SystemT2Cache{T}
    p::PT

    nls_state::NT
    r::Memory{T} # used to eval the objective function after solving BLS.
    w_nls::Memory{T}

    function NLSCostState(
        model::CLSurrogateModel{T},
        info::CLSurrogateInfo,
        num_fit_positions::Integer,
        nls_state::NT,
        ) where {T <: AbstractFloat, NT}

        design_matrix = Matrix{Complex{T}}(undef, num_fit_positions, get_num_compounds(model))

        p, _ = initialize_params(SystemT2(), info, model)
        state = SystemT2Cache(model, info)

        r = Memory{T}(undef, 2*num_fit_positions)
        fill!(r, T(Inf))

        w_nls = Memory{T}(undef, get_num_compounds(model))

        return new{T, typeof(p), NT}(design_matrix, state, p, nls_state, r, w_nls)
    end
end


# # Interface requirements
function get_concentration(s::AbstractNLSCostState)
    return s.w_nls
end

function get_model_params(s::AbstractNLSCostState)
    return s.p
end

function get_design_matrix(s::AbstractNLSCostState)
    return s.design_matrix
end

function get_model_state(s::AbstractNLSCostState)
    return s.model_state
end

function update_resonance_parameters!(S::AbstractNLSCostState, v::AbstractVector)
    return update_resonance_parameters!(S.p, v)
end

function update_concentration!(S::AbstractNLSCostState, w::AbstractVector)
    return update_concentration!(S.p, w)
end

function update_state!(S::AbstractNLSCostState, λ0::AbstractFloat, cbs::ResonanceGroupDCs)
    return update_state!(S.model_state, S.p, λ0, cbs)
end


# mutates `state`.
struct NLSCostCallable end
function (A!::NLSCostCallable)(state::NLSCostState, p::AbstractVector, params::NLSCostParameters)
    return eval_l2_cost_via_matrix!(state, p, params)
end

function eval_l2_cost_via_matrix!(
    S::NLSCostState{T}, # mutates. buffer.
    p_in::AbstractVector{T},
    ps::NLSCostParameters{T},
    ) where T
    
    nls_callable! = ps.nls_callable! # contains the nls params.
    nls_state = S.nls_state
    w_nls = S.w_nls

    update_resonance_parameters!(S, p_in)
    update_state!(S, ps.λ0, ps.Δc_bars)
    
    eval_cl_matrix!(
        S.design_matrix,
        ps.U_rad,
        ps.qs,
        S.model_state,
    )

    # # interlace real and imaginary parts of an entry as consecutive rows.
    # # We're assuming the nls_callable! does not work on complex-valued arrays.
    Br = reinterpret(T, S.design_matrix) # shouldn't allocate.
    #yr = reinterpret(T, ps.y) # shouldn't allocate.
    yr = ps.yr
    
    # # if using bls.
    # w_nls, _ = nls_callable!(nls_state, Br, yr) # the second slow for is status.
    nls_callable!(w_nls, nls_state, Br, yr, ps.nls_params)
    update_concentration!(S, w_nls)

    # compute cost
    r = S.r
    mul!(r, Br, w_nls) # 6 us
    r .= r .- yr
    return dot(r,r)
end

#### gradient and objective evaluation


struct NLSCostGradientState{T <: AbstractFloat, PT <: SurrogateParameters, NT} <: AbstractNLSCostState

    design_matrix::Matrix{Complex{T}}
    model_state::SystemT2Cache{T}
    p::PT

    nls_state::NT
    r::Memory{T} # used to eval the objective function after solving BLS.
    w_nls::Memory{T}

    df_dp::PT
    gr::PT
    gi::PT
    itp_evals::Matrix{Complex{T}}

    function NLSCostGradientState(
        model::CLSurrogateModel{T},
        info::CLSurrogateInfo,
        num_fit_positions::Integer,
        nls_state::NT,
        ) where {T <: AbstractFloat, NT}

        design_matrix = Matrix{Complex{T}}(undef, num_fit_positions, get_num_compounds(model))
        itp_evals = Matrix{Complex{T}}(undef, num_fit_positions, get_num_groups(model))

        p, _ = initialize_params(SystemT2(), info, model)
        gr, _ = initialize_params(SystemT2(), info, model)
        gi, _ = initialize_params(SystemT2(), info, model)
        df_dp, _ = initialize_params(SystemT2(), info, model)
        state = SystemT2Cache(model, info)

        r = Memory{T}(undef, 2*num_fit_positions)
        fill!(r, T(Inf))

        w_nls = Memory{T}(undef, get_num_compounds(model))

        return new{T, typeof(p), NT}(
            design_matrix, state, p, nls_state, r, w_nls, df_dp, gr, gi, itp_evals,
        )
    end
end

# mutates `state` as buffer, mutates df as output..
struct NLSCostGradientCallable end
function (A!::NLSCostGradientCallable)(df::AbstractVector, state::NLSCostGradientState, p::AbstractVector, params::NLSCostParameters)
    return eval_bls_cost_and_gradient!(df, state, p, params)
end

# after this function runs, g contain the gradient
function eval_bls_cost_and_gradient!(
    g::AbstractVector,
    S::NLSCostGradientState{T}, # mutates. buffer.
    p_in::AbstractVector{T},
    ps::NLSCostParameters{T},
    ) where T
    
    df_dp, itp_evals, gr, gi = S.df_dp, S.itp_evals, S.gr, S.gi
    length(g) == length(df_dp.resonance_contents) || error("Length mismatch.")

    nls_callable! = ps.nls_callable! # contains the nls params.
    nls_state = S.nls_state
    w_nls = S.w_nls

    update_resonance_parameters!(S, p_in)
    update_state!(S, ps.λ0, ps.Δc_bars)
    
    eval_cl_matrices!(
        S.design_matrix,
        itp_evals,
        ps.U_rad,
        ps.qs,
        S.model_state,
    )

    # # interlace real and imaginary parts of an entry as consecutive rows.
    # # We're assuming the nls_callable! does not work on complex-valued arrays.
    Br = reinterpret(T, S.design_matrix) # shouldn't allocate.
    #yr = reinterpret(T, ps.y) # shouldn't allocate.
    yr = ps.yr
    
    # # if using bls.
    # w_nls, _ = nls_callable!(nls_state, Br, yr) # the second slow for is status.
    nls_callable!(w_nls, nls_state, Br, yr, ps.nls_params)
    update_concentration!(S, w_nls)

    # compute cost
    r = S.r
    mul!(r, Br, w_nls) # 6 us
    r .= r .- yr
    out = dot(r,r)

    # # gradient
    eval_bls_gradient!(
        df_dp, gr, gi,
        S.model_state, itp_evals, ps.U_rad, w_nls, r,
        ps.qs, ps.Δc_bars, ps.λ0,
    )
    copy!(g, df_dp.resonance_contents)

    return out
end