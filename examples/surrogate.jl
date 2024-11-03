# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

include("a.jl")


#import PublicationDatasets as DS

include("helpers/utils.jl")

PLT.close("all")
fig_num = 1

#T = Float32
const T = Float64;

# # User inputs for simulation
# Let's use a preset from a BMRB experiment at 700 MHz.
fs, SW, ν_0ppm = HAM.getpresetspectrometer(T, "700");

# decay rate for the 0 ppm resonance component.
λ0 = convert(T, 3.4);

molecule_mapping_file_path = joinpath(pwd(), "ref_files", "molecule_name_mapping", "demo_compounds.json")
H_params_path = joinpath(pwd(), "ref_files", "coupling_info")

# list of target compounds for simulation and surrogate construction.
molecule_entries = [
    "alpha-D-Glucose";
    "beta-D-Glucose";
    "L-Serine";
    #"L-Leucine";
    #"L-Isoleucine";
    #"L-Valine";
    "DSS"; # this simulates the 0 ppm frequency reference compound.
    "Singlet - 4.9 ppm"; # this simulates D2O, the solvent.
];

# make up a set of relative concentration.
w_oracle = rand(T, length(molecule_entries))
w_oracle[end] = convert(T, 16.0); # make solvent very large.

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

# Complex Lorentzian surrogate configuration.
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

# FID surrogate configuration..
t_lb = zero(T)
t_ub = T(3)
fid_config = SIG.FIDSurrogateConfig{T}(
    Δt = T(1e-5),
    t_lb = t_lb,
    t_ub = t_ub,
)

println("Fitting surrogates. This might take a while.")
@time model, cl_info = SIG.create_cl_surrogate(As, λ0, cl_config);
@time fid_model, fid_info = SIG.create_fid_surrogate(As, λ0, fid_config);
# Bs and MSS are linked. Modification to one of its fields will affect the other.


# # Visualize
# Create parameter set, and parse out ws, the concentration portion.
p, mapping = SIG.initialize_params(SIG.SystemT2(), cl_info, model)
ws = p.concentration

# # Visualize with default parameters: zero phase, zero shift, all decay rate equal to λ0.
println()
println("Visualize with default parameters: zero phase, zero shift, all decay rate equal to λ0.")
include("plot.jl")

println("Now with randomized parameters")
rng = Random.Xoshiro(0)
randomize!(rng, p, κ_λ_lb, κ_λ_ub)
include("plot.jl")


nothing
