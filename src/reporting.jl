# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

#### Diagnostics and reporting

"""
    get_system_labels(As::Vector{SH}) where SH <: NMRHamiltonian.SHType

Outputs a `Vector{Tuple{Int,Int,Int}}` where each entry if of the form `(n,i,j)`, 
with `n` being the compound index, `i` being the spin system index, and `j` being the ME nucleui group index.
"""
function get_system_labels(As::Vector{SH}) where SH <: NMRHamiltonian.SHType
    rcs_nested, rcs_list = create_rcs_list(As)
    
    labels = Vector{Tuple{Int,Int,Int}}(undef, length(rcs_list))
    l = 0
    for n in eachindex(rcs_nested)
        for i in eachindex(rcs_nested[n])
            for j in eachindex(rcs_nested[n][i])
                l += 1
                labels[l] = (n,i,j)
            end
        end
    end

    return labels
end

function create_params_LUT(
    ::SystemT2,
    p::SurrogateParameters,
    mapping::ParameterMapping,
    As::Vector{SH},
    #compound_labels::AbstractVector{String}
    ) where SH <: NMRHamiltonian.SHType
    
    rcs_nested, rcs_list = create_rcs_list(As)
    contents = get_contents(p)
    cs_labels = get_system_labels(As)

    # shift vars
    inds = collect(mapping.shift)
    labels = cs_labels
    rcs = rcs_list
    vals = contents[inds]
    header = ["parameter index" "(compound, system, ME nuclei group)" "reference cs" "shift value"]
    shift_LUT = [header; inds labels rcs vals]

    # phase vars
    inds = collect(mapping.phase)
    labels = cs_labels
    rcs = rcs_list
    vals = contents[inds]
    header = ["parameter index" "(compound, system, ME nuclei group)" "reference cs" "phase value"]
    phase_LUT = [header; inds labels rcs vals]

    # κ_λ vars
    inds = collect(mapping.decay_multiplier)
    labels = collect( (x[1], x[2]) for x in cs_labels )
    labels = unique(labels)

    rcs = collect(Iterators.flatten(rcs_nested))
    vals = contents[inds]
    header = ["parameter index" "(compound, system)" "reference cs" "decay multiplier value"]
    κ_λ_LUT = [header; inds labels rcs vals]

    # concentration
    inds = collect(mapping.concentration)
    labels = collect( x[1] for x in cs_labels )
    labels = unique(labels)

    rcs = rcs_nested
    vals = contents[inds]
    header = ["parameter index" "compound" "reference cs" "relative concentration value"]
    w_LUT = [header; inds labels rcs vals]


    return shift_LUT, phase_LUT, κ_λ_LUT, w_LUT
end