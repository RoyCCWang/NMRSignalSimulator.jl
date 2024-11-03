# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

function parameter_gradients!(
    gr_buffer::SIG.SurrogateParameters, # mutates.
    gi_buffer::SIG.SurrogateParameters, # mutates.
    query_state, # mutates
    p_buffer::SIG.SurrogateParameters,
    p_range,
    u_rad::T,
    d::Integer,
    ) where T <: AbstractFloat

    q_ps = zeros(Complex{T}, length(p_range))
    grs = zeros(T, length(p_range))
    gis = zeros(T, length(p_range))

    # p_d_ref = SIG.get_contents(p_ref)[d]
    v_buffer = SIG.get_contents(p_buffer)

    for m in eachindex(p_range, q_ps, grs, gis)

        #query_state = SIG.create_state_reference(model, cl_info)
        v_buffer[d] = p_range[m]
        SIG.update_parameters!(p_buffer, v_buffer)
        SIG.update_state!(query_state, p_buffer)

        q_ps[m] = SIG.eval_cl_and_gradient!(
            gr_buffer,
            gi_buffer,
            query_state,
            u_rad,
        )
        grs[m] = SIG.get_contents(gr_buffer)[d]
        gis[m] = SIG.get_contents(gi_buffer)[d]
    end

    return q_ps, grs, gis
end

function parameter_gradients_ND!(
    gr,
    gi,
    v_buffer, # mutates
    query_state, # mutates
    p_range,
    u_rad::T,
    d::Integer,
    ) where T <: AbstractFloat

    q_U = zeros(Complex{T}, length(p_range))
    grs = zeros(T, length(p_range))
    gis = zeros(T, length(p_range))

    for m in eachindex(p_range, q_U, grs, gis)

        v_buffer[d] = p_range[m]

        # numerical derivatives
        hr = xx->real(SIG.eval_cl_and_gradient!(gr, gi, query_state, u_rad, xx))
        hi = xx->imag(SIG.eval_cl_and_gradient!(gr, gi, query_state, u_rad, xx))

        gr_ND = FiniteDiff.finite_difference_gradient(hr, v_buffer)
        gi_ND = FiniteDiff.finite_difference_gradient(hi, v_buffer)

        grs[m] = gr_ND[d]
        gis[m] = gi_ND[d]
        q_U[m] = Complex(hr(v_buffer), hi(v_buffer))
    end

    return q_U, grs, gis
end


function test_query_gradient!(
    gr::SIG.SurrogateParameters, # buffer, mutates
    gi::SIG.SurrogateParameters, # buffer, mutates
    query_state, # mutates
    p_ref::SIG.SurrogateParameters,
    u_rad::T,
    fc::SIG.FrequencyConversion,
    mapping;
    rel_discrepancy_tol = T(1e-5),
    Δppm_test_window = T(0.1),
    p_range_length = 1000,
    phase_lb = T(-2*π),
    phase_ub = T(2*π),
    decay_multiplier_lb = T(0.1),
    decay_multiplier_ub = T(6),
    concentration_lb = zero(T),
    concentration_ub = T(10),
    min_abs_discrepancy = T(1e-7), # if absolute descrepancy less than this, pass the test. Otherwise, compare the relative discrepancy against rel_discrepancy_tol.
    ) where T <: AbstractFloat

    for d in mapping.shift

        rad_window = abs(SIG.ppm_to_rad(zero(T), fc) - SIG.ppm_to_rad(Δppm_test_window, fc))
        p_range = LinRange(-rad_window, rad_window, p_range_length)

        q_U3, grs3, gis3 = parameter_gradients!(
            gr, gi, query_state, deepcopy(p_ref), p_range, u_rad, d,
        )
        q_U3_ND, grs3_ND, gis3_ND = parameter_gradients_ND!(
            gr, gi, copy(SIG.get_contents(p_ref)), query_state, p_range, u_rad, d,
        )
        
        disc = norm(q_U3 - q_U3_ND)
        if disc > min_abs_discrepancy
            @test norm(q_U3 - q_U3_ND)/norm(q_U3) < eps(T)
        end

        disc = norm(grs3 - grs3_ND)
        if disc > min_abs_discrepancy
            @test norm(grs3 - grs3_ND)/norm(grs3) < rel_discrepancy_tol
        end

        disc = norm(gis3 - gis3_ND)
        if disc > min_abs_discrepancy
            @test norm(gis3 - gis3_ND)/norm(gis3) < rel_discrepancy_tol
        end

    end

    for d in mapping.phase

        p_range = LinRange(phase_lb, phase_ub, p_range_length)

        q_U3, grs3, gis3 = parameter_gradients!(
            gr, gi, query_state, deepcopy(p_ref), p_range, u_rad, d,
        )
        q_U3_ND, grs3_ND, gis3_ND = parameter_gradients_ND!(
            gr, gi, copy(SIG.get_contents(p_ref)), query_state, p_range, u_rad, d,
        )

        disc = norm(q_U3 - q_U3_ND)
        if disc > min_abs_discrepancy
            @test norm(q_U3 - q_U3_ND)/norm(q_U3) < eps(T)
        end

        disc = norm(grs3 - grs3_ND)
        if disc > min_abs_discrepancy
            @test norm(grs3 - grs3_ND)/norm(grs3) < rel_discrepancy_tol
        end

        disc = norm(gis3 - gis3_ND)
        if disc > min_abs_discrepancy
            @test norm(gis3 - gis3_ND)/norm(gis3) < rel_discrepancy_tol
        end
        
    end

    for d in mapping.decay_multiplier

        p_range = LinRange(decay_multiplier_lb, decay_multiplier_ub, p_range_length)

        q_U3, grs3, gis3 = parameter_gradients!(
            gr, gi, query_state, deepcopy(p_ref), p_range, u_rad, d,
        )
        q_U3_ND, grs3_ND, gis3_ND = parameter_gradients_ND!(
            gr, gi, copy(SIG.get_contents(p_ref)), query_state, p_range, u_rad, d,
        )

        disc = norm(q_U3 - q_U3_ND)
        if disc > min_abs_discrepancy
            @test norm(q_U3 - q_U3_ND)/norm(q_U3) < eps(T)
        end

        disc = norm(grs3 - grs3_ND)
        if disc > min_abs_discrepancy
            @test norm(grs3 - grs3_ND)/norm(grs3) < rel_discrepancy_tol
        end

        disc = norm(gis3 - gis3_ND)
        if disc > min_abs_discrepancy
            @test norm(gis3 - gis3_ND)/norm(gis3) < rel_discrepancy_tol
        end
    
    end

    for d in mapping.concentration

        p_range = LinRange(concentration_lb, concentration_ub, p_range_length)

        q_U3, grs3, gis3 = parameter_gradients!(
            gr, gi, query_state, deepcopy(p_ref), p_range, u_rad, d,
        )
        q_U3_ND, grs3_ND, gis3_ND = parameter_gradients_ND!(
            gr, gi, copy(SIG.get_contents(p_ref)), query_state, p_range, u_rad, d,
        )

        disc = norm(q_U3 - q_U3_ND)
        if disc > min_abs_discrepancy
            @test norm(q_U3 - q_U3_ND)/norm(q_U3) < eps(T)
        end

        disc = norm(grs3 - grs3_ND)
        if disc > min_abs_discrepancy
            @test norm(grs3 - grs3_ND)/norm(grs3) < rel_discrepancy_tol
        end

        disc = norm(gis3 - gis3_ND)
        if disc > min_abs_discrepancy
            @test norm(gis3 - gis3_ND)/norm(gis3) < rel_discrepancy_tol
        end

    end

    return nothing
end