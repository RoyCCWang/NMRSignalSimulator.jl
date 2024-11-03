# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

function test_bls_gradient!(
    rng, # mutates
    p_oracle,
    model,
    cl_info,
    κ_λ_lb::T,
    κ_λ_ub::T,
    κ_ζ_lb::T,
    κ_ζ_ub::T,
    N_tests::Integer,
    w_lbs::AbstractVector{T};
    zero_tol = eps(T)*10,
    min_abs_discrepancy = T(1e-12),
    rel_discrepancy_tol = T(1e-4),
    ) where T <: AbstractFloat
    
    if min_abs_discrepancy == zero(T)
        min_abs_discrepancy = eps(T)*100
    end

    # compute oracle observations
    query_state = SIG.create_state_reference(model, cl_info)
    SIG.update_state!(query_state, p_oracle)

    hz_lb, hz_ub = SIG.get_frequency_bounds(cl_info)
    U_hz = LinRange(hz_lb, hz_ub, 10000)
    U_rad = Memory{T}(U_hz .* T(2*π))
    y = collect( SIG.eval_cl(query_state, u_rad) for u_rad in U_rad )

    # setup cost function, f
    cost_state = SIG.NLSCostState(model, cl_info, length(y), NNLSState(T, length(y)))
    cost_params = SIG.NLSCostParameters(
        U_hz, y, model, NNLSParameters(w_lbs), NNLSCallable(),
    )
    cost_func! = SIG.NLSCostCallable()
    f = vv->cost_func!(cost_state, vv, cost_params)

    # setup cost and gradient function, fdf!
    fdf_state = SIG.NLSCostGradientState(model, cl_info, length(y), NNLSState(T, length(y)))
    fdf_params = cost_params
    fdf! = SIG.NLSCostGradientCallable()


    # # test

    # pre-allocate
    p_test, _ = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
    df_eval = Memory{T}(SIG.get_resonance_contents(p_test))
    

    for _ = 1:N_tests
        fill!(df_eval, T(NaN))
        randomize!(rng, p_test, κ_λ_lb, κ_λ_ub, κ_ζ_lb, κ_ζ_ub)

        v_test = SIG.get_resonance_contents(p_test)
        out = cost_func!(cost_state, v_test, cost_params)
        out2 = fdf!(df_eval, fdf_state, v_test, fdf_params)

        @assert abs(out-out2) < zero_tol

        # check against numerical gradient
        df_eval_ND = FiniteDiff.finite_difference_gradient(f, v_test)

        abs_disc = norm(df_eval - df_eval_ND)
        if abs_disc > min_abs_discrepancy
            
            @test abs_disc/norm(df_eval) < rel_discrepancy_tol
        end
    end

    return nothing
end

# mutates rng.
function test_bls_gradient!(
    rng, model, cl_info,
    κ_λ_lb::T, κ_λ_ub::T,
    κ_ζ_lb::T, κ_ζ_ub::T,
    N_oracles::Integer, N_tests::Integer;
    zero_tol = eps(T)*10,
    min_abs_discrepancy = T(1e-12),
    rel_discrepancy_tol = T(1e-4),
    ) where T

    for _ = 1:N_oracles
        
        # generate oracles
        p_oracle, _ = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
        randomize!(rng, p_oracle, κ_λ_lb, κ_λ_ub)
        
        w_lbs = rand(T, SIG.get_num_compounds(model))

        # test the gradient against the oracle-generated cost func.
        test_bls_gradient!(
            rng,
            p_oracle, model, cl_info,
            κ_λ_lb, κ_λ_ub,
            κ_ζ_lb, κ_ζ_ub,
            N_tests, w_lbs;
            zero_tol,
            min_abs_discrepancy,
            rel_discrepancy_tol,
        )
    end

    return nothing
end