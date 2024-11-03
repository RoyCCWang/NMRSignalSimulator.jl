# SPDX-License-Identifier: GPL-3.0-only
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

"""
    convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T

converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
"""
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generateparameters(lbs::Vector{T}, ubs::Vector{T})::Vector{T} where T
    
    @assert length(lbs) == length(ubs)

    return collect( convertcompactdomain(rand(), zero(T), one(T), lbs[i], ubs[i]) for i in eachindex(lbs) )
end

function combinevectors(x::Vector{Vector{T}})::Vector{T} where T

    if isempty(x)
        return Vector{T}(undef, 0)
    end

    N = sum(length(x[i]) for i in eachindex(x))

    y = Vector{T}(undef,N)

    st_ind = 0
    fin_ind = 0
    for i in eachindex(x)
        st_ind = fin_ind + 1
        fin_ind = st_ind + length(x[i]) - 1

        y[st_ind:fin_ind] = x[i]
    end

    return y
end

function randomize!(rng, p::SIG.SurrogateParameters, κ_λ_lb::T, κ_λ_ub) where T <: AbstractFloat
    κ_λ_lb < κ_λ_ub || erorr("κ_λ_lb must be smaller than κ_λ_ub.")

    for i in eachindex(p.shift)
        randn!(rng, p.shift[i])
        #fill!(p.shift[i], 4*2199.0)
    end

    for i in eachindex(p.phase)
        randn!(rng, p.phase[i])
    end

    for i in eachindex(p.decay_multiplier)
        p.decay_multiplier[i] = convertcompactdomain(rand(rng, T), zero(T), one(T), κ_λ_lb, κ_λ_ub)
    end

    rand!(rng, p.concentration)

    return nothing
end

function randomize!(rng, p::SIG.SurrogateParameters, κ_λ_lb::T, κ_λ_ub, κ_ζ_lb::T, κ_ζ_ub::T) where T <: AbstractFloat
    κ_λ_lb < κ_λ_ub || erorr("κ_λ_lb must be smaller than κ_λ_ub.")
    κ_ζ_lb < κ_ζ_ub || erorr("κ_ζ_lb must be smaller than κ_ζ_ub.")
    
    for i in eachindex(p.shift)
        for j in eachindex(p.shift[i])
            p.shift[i][j] = convertcompactdomain(rand(rng, T), zero(T), one(T), κ_ζ_lb, κ_ζ_ub)
        end
    end

    for i in eachindex(p.phase)
        randn!(rng, p.phase[i])
    end

    for i in eachindex(p.decay_multiplier)
        p.decay_multiplier[i] = convertcompactdomain(rand(rng, T), zero(T), one(T), κ_λ_lb, κ_λ_ub)
    end

    rand!(rng, p.concentration)

    return nothing
end