# Copyright (c) 2025-2026 Jan Philipp Thiele
# SPDX-License-Identifier: MIT

"""
   PCE

This submodule provides all the functionality for building a PCE surrogate from an FEM sample obtained through the `Sampling` submodule.
"""
module PCE

using ..Sampling: UnivariateFEMSample
import ..Sampling: compute_statistics

using Statistics
import Statistics: mean, var, cov

using PolyChaos: AbstractCanonicalOrthoPoly, GaussOrthoPoly, evaluate, computeSP2
using Distributions: quantile, UnivariateDistribution, Normal

export PolyChaosExpansion
export mean

"""
    PolyChaosExpansion{T<: Real}

Struct for holding the PCE surrogate
"""
struct PolyChaosExpansion{T <: Real}
    coefficients::AbstractMatrix{T}
    orthogonal_polynomials::AbstractCanonicalOrthoPoly
end


"""
    PolyChaosExpansion(sample::UnivariateFEMSample{T};polynomials::AbstractCanonicalOrthoPoly = GaussOrthoPoly(8),vector_distribution::UnivariateDistribution=Normal(0,1)) where T<:Real

Represent the sampled random variable ``Y``(`sample`) as a Polynomial Chaos Expansion (PCE) surrogate
`` Y = ∑_{i∈ℕ} cᵢΨᵢ(𝐗) ``
with: 
- ``Ψᵢ`` being a polynomial basis function from `polynomials`, default gaussion/probabilists' Hermite of degree 8
- ``𝐗`` a random vector of distribution `vector_distribution`, default ``𝐗∼𝒩(0,1)``,
- ``cᵢ`` the PCE coefficients to be calculated.
"""
function PolyChaosExpansion(sample::UnivariateFEMSample{T}; polynomials::AbstractCanonicalOrthoPoly = GaussOrthoPoly(8), vector_distribution::UnivariateDistribution = Normal(0, 1)) where {T <: Real}
    n = length(sample.uniform_sample)
    @assert n >= 2 + 3 * (polynomials.deg + 1) "Too few samples.\n Use at least (2 + 3 * (polynomial_degree + 1)) for numerical stability"
    coeff = calculate_PCE_coefficients(quantile(vector_distribution, sample.uniform_sample), sample.fem_sample, polynomials)
    return PolyChaosExpansion{T}(coeff, polynomials)
end


"""
    calculate_PCE_coefficients(x::AbstractVector{T}, Y::AbstractMatrix{T}, orthopolys::AbstractCanonicalOrthoPoly) where T<:Real

Calculate the coefficients of a polynomial chaos expansion based on measurements Y for parameters x
"""
function calculate_PCE_coefficients(x::AbstractVector{T}, Y::AbstractMatrix{T}, orthopolys::AbstractCanonicalOrthoPoly) where {T <: Real}
    n = length(x)
    A = zeros(T, n, orthopolys.deg + 1)
    for i in eachindex(x)
        A[i, :] = evaluate(x[i], orthopolys)
    end
    return (A' * A) \ (A' * Y)
end

"""
    mean(pce::PolyChaosExpansion{T})::AbstractVector{T} where T<:Real

Compute mean values of the given PCE
"""
function Statistics.mean(pce::PolyChaosExpansion{T})::AbstractVector{T} where {T <: Real}
    return vec(pce.coefficients[1, :])
end


"""
    var(pce::PolyChaosExpansion{T})::AbstractVector{T} where T <: Real

Compute variance of the given PCE
"""
function Statistics.var(pce::PolyChaosExpansion{T})::AbstractVector{T} where {T <: Real}
    normsq = computeSP2(pce.orthogonal_polynomials)
    result = zeros(T, size(pce.coefficients)[2])
    for i in eachindex(normsq)[2:end]
        result += (pce.coefficients[i, :] .^ 2 * normsq[i])
    end
    return result
end

"""
    cov(pce::PolyChaosExpansion{T})::AbstractMatrix{T} where T <: Real

Compute cov matrix of the given PCE
"""
function Statistics.cov(pce::PolyChaosExpansion{T})::AbstractMatrix{T} where {T <: Real}
    normsq = computeSP2(pce.orthogonal_polynomials)
    result = zeros(T, size(pce.coefficients)[2], size(pce.coefficients)[2])
    for i in eachindex(normsq)[2:end]
        result += pce.coefficients[i, :] * transpose(pce.coefficients[i, :]) * normsq[i]
    end
    return result
end

"""
    std(pce::PolyChaosExpansion{T})::AbstractVector{T} where T <: Real

Compute standard deviation of the given PCE
"""
function std(pce::PolyChaosExpansion{T})::AbstractVector{T} where {T <: Real}
    return sqrt.(var(pce))
end

"""
    compute_statistics(pce::PolyChaosExpansion)

Compute mean and standard deviation of the given PCE
"""
function compute_statistics(pce::PolyChaosExpansion)
    return mean(pce), std(pce)
end

end
