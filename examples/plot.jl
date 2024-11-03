


# FID
t_range = LinRange(t_lb, t_ub, 10000)

fid_query_state = SIG.create_state_reference(fid_model, fid_info)
SIG.update_state!(fid_query_state, p)
surrogate_t = collect( SIG.eval_fid(fid_query_state, t) for t in t_range )

prelim_t = collect( SIG.eval_fid(t, ws, As, SIG.get_prelim_state(fid_query_state)) for t in t_range )


println("discrepancy to prelim vs. surrogate for free-induction decay model:")
@show norm(prelim_t - surrogate_t)/norm(prelim_t)
println()

PLT.figure(fig_num)
fig_num += 1
PLT.plot(t_range, real.(prelim_t), label = "prelim")
PLT.plot(t_range, real.(surrogate_t), label = "surrogate", "--")
PLT.gca().invert_xaxis() #hide
PLT.xlabel("Chemical shift δ (ppm)")
PLT.ylabel("Real part")
PLT.legend()
PLT.title("Free-induction decay models: prelim vs. surrogate")

# Complex Lorentzian

U = LinRange(cl_info.hz_lb, cl_info.hz_ub, 10000)
U_rad = Memory{T}(U .* T(2*π))
fc = SIG.get_frequency_conversion(model)
P = collect( SIG.hz_to_ppm(u, fc) for u in U )

# Note that eval_cl takes in frequency in radians, not Hz, not ppm.
cl_query_state = SIG.create_state_reference(model, cl_info)
SIG.update_state!(cl_query_state, p)

prelim_U = collect( SIG.eval_cl(u_rad, ws, As, SIG.get_prelim_state(cl_query_state)) for u_rad in U_rad )
surrogate_U = collect( SIG.eval_cl(cl_query_state, u_rad) for u_rad in U_rad )

println("discrepancy to prelim vs. surrogate for complex Lorentzian model:")
@show norm(prelim_U - surrogate_U)/norm(prelim_U)
println()

PLT.figure(fig_num)
fig_num += 1
PLT.plot(P, real.(prelim_U), label = "prelim")
PLT.plot(P, real.(surrogate_U), label = "surrogate", "--")
PLT.legend()
PLT.title("Complex Lorentzian models: prelim vs. surrogate")

nothing