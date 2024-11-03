# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

using Test, LinearAlgebra

import NMRHamiltonian as HAM
import NMRSignalSimulator as SIG
import FiniteDiff

include("../examples/helpers/utils.jl")
include("./helpers/query_derivatives.jl")

# for BLS gradient.
import NonNegLeastSquares as NNLS
include("./helpers/bls_derivatives.jl")
include("../examples/helpers/nls.jl")



using Random
Random.seed!(25)

# based on cl_fit_objective.jl and gradient.jl
@testset "query derivatives" begin

    rng = Random.Xoshiro(0)
    T = Float64

    # # User inputs for simulation
    # Let's use a preset from a BMRB experiment at 700 MHz.
    fs, SW, ν_0ppm = HAM.getpresetspectrometer(T, "700");

    # Specify a 1/T2 inverse time constant.
    λ0 = convert(T, 3.4);

    # put the database coupling values into dictionary structures. You can supply your own coupling values; see the documentation website for NMRSignalSimulator.jl and NMRHamiltonian.jl.
    molecule_mapping_file_path = joinpath(dirname(pwd()), "examples", "ref_files", "molecule_name_mapping", "demo_compounds.json")  
    H_params_path = joinpath(dirname(pwd()), "examples", "ref_files", "coupling_info")

    # list of target compounds for simulation and surrogate construction.
    molecule_entries = [
        #"alpha-D-Glucose";
        #"beta-D-Glucose";
        "Singlet - 0 ppm";
        "L-Serine";
        #"DSS"; # this simulates the 0 ppm frequency reference compound.
        "Singlet - 4.9 ppm"; # this simulates D2O, the solvent.
    ];

    # # Spin Hamiltonian simulation and resonance group computation
    # generate the spin Hamiltonian simulation and cluster to get resonance groups. See NMRHamiltonian.jl documentation for further details for the following code.
    config = HAM.SHConfig{T}(
        coherence_tol = convert(T, 0.01),
        relative_α_threshold = convert(T, 0.001),
        max_deviation_from_mean = convert(T, 0.05),
        acceptance_factor = convert(T, 0.99),
        total_α_threshold = zero(T),
    )
    unique_cs_digits = 6

    println("Timing: HAM.loadandsimulate")
    @time Phys, As, MSPs = HAM.loadandsimulate(
        fs, SW, ν_0ppm,
        molecule_entries,
        H_params_path,
        molecule_mapping_file_path,
        config;
        unique_cs_digits = unique_cs_digits,
    );

    # # Frequency-domain surrogate construction
    κ_λ_lb = T(0.5) # lower limit for κ_λ for which the surrogate is made from.
    κ_λ_ub = T(2.5) # upper limit for κ_λ for which the surrogate is made from.
    Δppm_border = T(0.5)  # In units of ppm. interpolation border that is added to the lowest and highest resonance frequency component of the mixture being simulated.

    cl_config = SIG.CLSurrogateConfig{T}(
        #Δr = convert(T, 1.0), # radial frequency resolution: smaller means slower to build surrogate, but more accurate.
        Δr = convert(T, 1.0),
        Δκ_λ = convert(T, 0.05), # T2 multiplier resolution. smaller means slower to build surrogate, but more accurate.
        κ_λ_lb = κ_λ_lb,
        κ_λ_ub = κ_λ_ub,
        Δppm_border = Δppm_border,
    )

    t_lb = zero(T)
    t_ub = T(3)
    fid_config = SIG.FIDSurrogateConfig{T}(
        Δt = T(1e-5),
        t_lb = t_lb,
        t_ub = t_ub,
    )

    println("Timing: SIG.create_cl_surrogate")
    @time model, cl_info = SIG.create_cl_surrogate(As, λ0, cl_config);
    @time fid_model, fid_info = SIG.create_fid_surrogate(As, λ0, fid_config);
    # Bs and MSS are linked. Modification to one of its fields will affect the other.

    # visualize surrogate

    state = SIG.SystemT2Cache(model, cl_info) # create state from model.

    # generate the oracle parameters.
    p, mapping = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
    randomize!(rng, p, κ_λ_lb, κ_λ_ub) # generate parameters. see examples/helpers/utils.jl

    # load parameters to model. This computes ζ, λ, β for each resonance group.
    SIG.update_state!(state, p, model)
    ws_oracle = p.concentration

    # CL

    hz_lb, hz_ub = SIG.get_frequency_bounds(cl_info)
    U = LinRange(hz_lb, hz_ub, 10000)
    U_rad = Memory{T}(U .* T(2*π))
    fc = SIG.get_frequency_conversion(model)
    P = collect( SIG.hz_to_ppm(u, fc) for u in U )

    out_prelim = collect( SIG.eval_cl(u_rad, ws_oracle, As, state) for u_rad in U_rad )

    query_state = SIG.create_state_reference(model, cl_info)
    SIG.update_state!(query_state, p)

    # # cl surrogate, query gradient

    # gradient buffers for the real and imaginary parts.
    gr, _ = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
    gi, _ = SIG.initialize_params(SIG.SystemT2(), cl_info, model)

    # test frequency position.
    u_rad_test = U_rad[842] # around the 0 ppm peak.

    # test parameter, pre-allocate.
    p_test, mapping = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
    rng_test = Xoshiro(1)
    
    # each test randomizes within range of the decay multiplier.
    N_tests = 20
    for _ = 1:N_tests
        randomize!(rng_test, p_test, κ_λ_lb, κ_λ_ub) # generate parameters. see examples/helpers/utils.jl
        test_query_gradient!(
            gr, gi, query_state,
            p_test,
            u_rad_test, fc, mapping,
        )
    end

    # # bounded least squares gradient, envelop theorem.

    κ_ζ_lb = -SIG.Δppm_to_Δrad(T(0.5), fc)
    κ_ζ_ub = SIG.Δppm_to_Δrad(T(0.5), fc)

    N_tests = 10
    N_oracles = 10

    test_bls_gradient!(
        Random.Xoshiro(100),
        model, cl_info,
        T(0.5)*κ_λ_lb, T(1.5)*κ_λ_ub,
        T(0.5)*κ_ζ_lb, T(1.5)*κ_ζ_ub,
        N_oracles, N_tests;
        rel_discrepancy_tol = T(1e-4),
    )

end
