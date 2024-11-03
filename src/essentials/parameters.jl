# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

abstract type T2Option end
struct SystemT2 <: T2Option end

function get_num_params(::SystemT2, cbs::ResonanceGroupDCs)

    N_shift = get_num_coherence_params(cbs)
    N_phase = N_shift # length(s.cos_βs)
    N_T2 = get_num_sys(cbs)
    N_compounds = length(cbs.nested)

    return N_shift, N_phase, N_T2, N_compounds
end

function get_total_num_params(::SystemT2, cbs::ResonanceGroupDCs)
    return sum(get_num_params(SystemT2(), cbs))
end

function get_num_coherence_params(cbs::ResonanceGroupDCs)
    Δc_bars = get_nested(cbs)
    N = 0
    for n in eachindex(Δc_bars)
        for i in eachindex(Δc_bars[n])
            N += size(Δc_bars[n][i], 2)
        end
    end
    return N
end

#const ParameterSubArray{T} = SubArray{T, 1, Memory{T}, Tuple{UnitRange{Int}}, true}
function get_dummy_mem_subarray_type(::Type{T}) where T
    a = Memory{T}(undef, 3)
    return typeof(view(a, 1:2))
end

struct SurrogateParameters{T <: Real, ST <: SubArray}
    contents::Memory{T}
    # shift::Memory{ParameterSubArray{T}} # each index is for a resonance group.
    # phase::Memory{ParameterSubArray{T}} # each index is for a resonance group.
    # decay_multiplier::ParameterSubArray{T} # each index is for a spin system (i.e. use with SystemT2) or for a resonance group (GroupT2)
    # concentration::ParameterSubArray{T} # each index is for a compound.
    shift::Memory{ST} # each index is for a resonance group.
    phase::Memory{ST} # each index is for a resonance group.
    decay_multiplier::ST # each index is for a spin system (i.e. use with SystemT2) or for a resonance group (GroupT2)
    concentration::ST # each index is for a compound.
    resonance_contents::ST # without concentration.
end

function get_contents(p::SurrogateParameters)
    return p.contents
end

# contents without concentration.
function get_resonance_contents(p::SurrogateParameters)
    return p.resonance_contents
end

function get_concentration(p::SurrogateParameters)
    return p.concentration
end

function update_parameters!(p::SurrogateParameters, v::AbstractVector)
    copy!(p.contents, v)
    return nothing
end

function update_resonance_parameters!(p::SurrogateParameters, v::AbstractVector)
    pv = get_resonance_contents(p)
    copy!(pv, v)
    return nothing
end

function update_concentration!(p::SurrogateParameters, w::AbstractVector)
    pv = get_concentration(p)
    copy!(pv, w)
    return nothing
end


struct ParameterMapping{RT}
    shift::RT
    phase::RT
    decay_multiplier::RT
    concentration::RT
end

# assumes the ranges in `mapping` are consecutive, with `mapping.shift[begin]` taking the value 1. and in ascending order such that `mapping.concentration[end]` is the last index.
function get_total_num_params(mapping::ParameterMapping)
    return mapping.concentration[end]
end

function get_total_num_resonance_params(mapping::ParameterMapping)
    return mapping.decay_multiplier[end]
end

# assumes the ranges in `mapping` are consecutive, with `mapping.shift[begin]` taking the value 1. and in ascending order such that `mapping.concentration[end]` is the last index.
function create_bounds_array(
    mapping::ParameterMapping,
    shift_ub::T,
    phase_lb::T, phase_ub::T,
    decay_multiplier_lb::T, decay_multiplier_ub::T,
    concentration_lb::T, concentration_ub::T,
    ) where T

    N = get_total_num_params(mapping)
    lbs = Memory{T}(undef, N)
    ubs = Memory{T}(undef, N)
    
    for i in mapping.shift
        lbs[i] = -shift_ub
        ubs[i] = shift_ub
    end

    for i in mapping.phase
        lbs[i] = phase_lb
        ubs[i] = phase_ub
    end

    for i in mapping.decay_multiplier
        lbs[i] = decay_multiplier_lb
        ubs[i] = decay_multiplier_ub
    end

    for i in mapping.concentration
        lbs[i] = concentration_lb
        ubs[i] = concentration_ub
    end

    return lbs, ubs
end

function create_bounds_array(
    mapping::ParameterMapping,
    shift_ub::T,
    phase_lb::T, phase_ub::T,
    decay_multiplier_lb::T, decay_multiplier_ub::T,
    ) where T

    N = get_total_num_resonance_params(mapping)
    lbs = Memory{T}(undef, N)
    ubs = Memory{T}(undef, N)
    
    for i in mapping.shift
        lbs[i] = -shift_ub
        ubs[i] = shift_ub
    end

    for i in mapping.phase
        lbs[i] = phase_lb
        ubs[i] = phase_ub
    end

    for i in mapping.decay_multiplier
        lbs[i] = decay_multiplier_lb
        ubs[i] = decay_multiplier_ub
    end

    return lbs, ubs
end

function get_num_params(option::T2Option, model::CLSurrogateModel)
    return get_num_params(option, get_Δc(model))
end

# default method is to treat concentrationsas parameters.
function initialize_params(
    option::T2Option,
    model_info::SurrogateInfo,
    model::SurrogateModel{T};
    initial_concentration::T = one(T),
    ) where T <: AbstractFloat
    
    rcs_nested = get_rcs(model_info)
    cbs = get_Δc(model)
    N_shift, N_phase, N_T2, N_compounds = get_num_params(option, cbs)

    # initialize content
    contents = Memory{T}(undef, N_shift + N_phase + N_T2 + N_compounds)
    fill!(contents, Inf) # TODO replace with unit tests.
    
    st_ind = 1
    fin_ind = st_ind + N_shift - 1
    shift = view(contents, st_ind:fin_ind)
    fill!(shift, zero(T))
    shift_range = st_ind:fin_ind

    st_ind = fin_ind + 1
    fin_ind = st_ind + N_phase - 1
    phase = view(contents, st_ind:fin_ind)
    fill!(phase, zero(T))
    phase_range = st_ind:fin_ind

    st_ind = fin_ind + 1
    fin_ind = st_ind + N_T2 - 1
    decay_multiplier = view(contents, st_ind:fin_ind)
    fill!(decay_multiplier, 1)
    decay_range = st_ind:fin_ind

    st_ind = fin_ind + 1
    fin_ind = st_ind + N_compounds - 1
    ws = view(contents, st_ind:fin_ind)
    fill!(ws, initial_concentration)
    concentration_range = st_ind:fin_ind

    return create_views(option, contents, rcs_nested, cbs), ParameterMapping(shift_range, phase_range, decay_range, concentration_range)
end

mutable struct StartFinish
    st_ind::Int
    fin_ind::Int
    function StartFinish()
        return new(0,0)
    end
end

function make_range!(A::StartFinish, L::Integer)
    A.st_ind = A.fin_ind + 1
    A.fin_ind = A.st_ind + L - 1
    
    st = A.st_ind
    fin = A.fin_ind
    return st:fin
end

function create_views(option::T2Option, contents::Memory{T}, rcs_nested, cbs::ResonanceGroupDCs{T}) where T <: AbstractFloat

    N_shift, N_phase, N_T2, N_compounds = get_num_params(option, cbs)
    (length(contents) == N_shift + N_phase + N_T2 + N_compounds) || error("Parameter array has incorrect length")

    ST = get_dummy_mem_subarray_type(T)

    # create views.
    #Δc_bars_nested = cbs.nested
    st_ind = 0
    fin_ind = 0

    Q = StartFinish()
    views_shift = Memory{ST}(
        collect(
            view(contents, make_range!(Q, length(r_ni)))
            for r_ni in Iterators.flatten(rcs_nested)
        )
    )

    views_phase = Memory{ST}(
        collect(
            view(contents, make_range!(Q, length(r_ni)))
            for r_ni in Iterators.flatten(rcs_nested)
        )
    )

    st_ind, fin_ind = Q.st_ind, Q.fin_ind
    st_ind = fin_ind + 1
    fin_ind = st_ind + N_T2 -1
    views_T2 = view(contents, st_ind:fin_ind)

    views_rc = view(contents, 1:fin_ind)

    # The remainder is for T2.
    st_ind = fin_ind + 1
    fin_ind = st_ind + N_compounds -1
    views_concentration = view(contents, st_ind:fin_ind)

    # sanity check.
    length(contents) == fin_ind || error("Index count error, please file a bug report.")

    return SurrogateParameters(
        contents, views_shift, views_phase, views_T2, views_concentration, views_rc,
    )
end

# mutates `s`.
function update_state!(s::SystemT2Cache, p::SurrogateParameters, model::SurrogateModel)
    return update_state!(s, p, get_λ0(model), get_Δc(model))
end

function update_state!(
    s::SystemT2Cache, # mutates.
    p::SurrogateParameters,
    λ0::T,
    cbs::ResonanceGroupDCs,
    ) where T <: AbstractFloat

    cbr = get_nested(cbs)
    shift, phase, decay_multiplier = p.shift, p.phase, p.decay_multiplier
    rg, λs = s.rg, s.λs

    ζ_buf = s.ζ_buffer
    β_buf = s.β_buffer

    l = 0
    for n in eachindex(rg, cbr, ζ_buf, β_buf)
        rg_n = rg[n]
        cbr_n = cbr[n]
        ζ_buf_n = ζ_buf[n]
        β_buf_n = β_buf[n]

        for i in eachindex(rg_n, cbr_n, ζ_buf_n, β_buf_n)
            rg_ni = rg_n[i]
            ζ_buf_ni = ζ_buf_n[i]
            β_buf_ni = β_buf_n[i]

            l += 1
            mul!(ζ_buf_ni, cbr_n[i], shift[l])
            mul!(β_buf_ni, cbr_n[i], phase[l])
            

            for k in eachindex(rg_ni, ζ_buf_ni, β_buf_ni)
                rg_ni[k] = RGState(ζ_buf_ni[k], cis(β_buf_ni[k]))
            end
        end
    end
    
    # T2: one scalar per spin system.
    l = 0
    for λ_n in λs
        for i in eachindex(λ_n)
            l += 1
            λ_n[i] = λ0*decay_multiplier[l]
        end
    end
    return nothing
end

###### For identifying the subset of model parameters within a given chemical shift interval.

function create_rcs_list(As::Vector{NMRHamiltonian.SHType{T}}) where T <: AbstractFloat

    #rcs_nested = deepcopy(get_nested(cbr))
    rcs_nested = Memory{Memory{Memory{T}}}(undef, length(As))

    for n in eachindex(As, rcs_nested)
        rcs_nested[n] = Memory{Memory{T}}(undef, length(As[n].Δc_bar))
        
        rcs_src = NMRHamiltonian.get_rcs(As[n])
        for i in eachindex(rcs_nested[n], rcs_src)
            rcs_nested[n][i] = copy(rcs_src[i])
        end
    end

    return rcs_nested, flatten_rcs(rcs_nested)
end

function flatten_rcs(rcs_nested::Memory{Memory{Memory{T}}}) where T
    rcs_flat = Memory{T}(
        collect(Iterators.flatten(Iterators.flatten(rcs_nested)))
    )
    return rcs_flat
end

struct ActiveIndices
    shift::Memory{Int}
    phase::Memory{Int}
    decay_multiplier::Memory{Int}
    concentration::Memory{Int}
end

function combine_indices(A::ActiveIndices)
    return sort(unique([A.shift; A.phase; A.decay_multiplier; A.concentration]))
end

function find_parameters(
    ::SystemT2,
    model_info::SurrogateInfo,
    model::SurrogateModel,
    cs_lb::T,
    cs_ub::T,
    Δppm_border::T,
    ) where T <: AbstractFloat

    cs_lb < cs_ub || error("Invalid chemical shift bounds.")

    # Get the list of reference chemical shifts.
    rcs_nested = get_rcs(model_info)
    rcs_flat = flatten_rcs(rcs_nested)
    N_dc = length(rcs_flat)

    cbs = get_Δc(model)
    N_shift, N_phase, N_T2, N_compounds = get_num_params(SystemT2(), cbs)
    N_shift == N_dc || error("Incorrect number of shift parameters.")
    N_phase == N_dc || error("Incorrect number of phase parameters.")
    N_compounds == length(rcs_nested) || error("Incorrect number of compounds.")
    
    # resultant rcs that fall within the [cs_lb,cs_ub] interval, perturned by Δppm_border is:
    flags = falses(N_dc)
    for i in eachindex(rcs_flat, flags)
        if cs_lb <= rcs_flat[i] + Δppm_border && rcs_flat[i] - Δppm_border <= cs_ub
            flags[i] = true
        end
    end
    inds_dc = (1:N_dc)[flags]

    # for system decay.
    T2_flags = falses(N_T2)
    w_flags = falses(N_compounds)
    j = 0
    for n in eachindex(rcs_nested)
        for i in eachindex(rcs_nested[n])
            j += 1

            test_cs = minimum(rcs_nested[n][i])
            if cs_lb <= test_cs + Δppm_border && test_cs - Δppm_border <= cs_ub
                T2_flags[j] = true
                w_flags[n] = true
            end

            test_cs = maximum(rcs_nested[n][i])
            if cs_lb <= test_cs + Δppm_border && test_cs - Δppm_border <= cs_ub
                T2_flags[j] = true
                w_flags[n] = true
            end
        end
    end
    inds_T2 = (1:N_T2)[T2_flags]
    inds_w = (1:N_compounds)[w_flags]
    
    # For SystemT2:
    # first N_dc entries are for shift.
    # next N_dc entries are for phase.

    active_inds_shift = inds_dc
    st_ind = N_shift

    active_inds_phase = inds_dc .+ st_ind
    st_ind += N_phase

    active_inds_T2 = inds_T2 .+ st_ind
    st_ind += N_T2

    active_inds_w = inds_w .+ st_ind

    #return sort(unique(active_inds_shift; active_inds_phase; active_inds_T2; active_inds_w))
    return ActiveIndices(
        Memory{Int}(active_inds_shift),
        Memory{Int}(active_inds_phase),
        Memory{Int}(active_inds_T2),
        Memory{Int}(active_inds_w),
    )
end