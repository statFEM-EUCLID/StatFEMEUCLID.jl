# Copyright (c) 2025-2026 Jan Philipp Thiele
# SPDX-License-Identifier: MIT


"""
    Sampling

This submodule provides all the high-level methods to sample the black box model through UM-Bridge.
"""
module Sampling
import ..FEMClient: evaluate_fem_model, empty_config

export UnivariateFEMSample, sample_FEM, compute_statistics

using StatsBase: mean, std
using UMBridge: HTTPModel, model_output_sizes
using Distributions: Normal, UnivariateDistribution, quantile, params
using Random: default_rng
using Mocking: @mock

"""
    struct UnivariateFEMSample{T<:Real}

Stores the result of a sampling call, i.e. the underlying uniform sample, the actual sample distribution
and the obtained FEM sample
"""
struct UnivariateFEMSample{T <: Real}
    uniform_sample::AbstractVector{T}
    sample_distribution::UnivariateDistribution
    fem_sample::AbstractMatrix{T}
end

function draw_FEM_samples(fem_model::HTTPModel, parameter_sample::Vector{Float64}; solution_index = 1, config = empty_config())
    samples = zeros(length(parameter_sample), @mock(model_output_sizes(fem_model))[1])
    for i in eachindex(parameter_sample)
        samples[i, :] .= evaluate_fem_model(fem_model, parameter_sample[i], solution_index = solution_index, config = config)
    end
    return samples
end

"""
    sample_FEM(fem_model::HTTPModel, n_samples::Int;sample_distribution::UnivariateDistribution,rng = default_rng(),solution_index=1,config=empty_config())::UnivariateFEMSample

Sample the `fem_model` `n_samples` times with a univariate random parameter with distribution `sample_distribution`.

Optional keyword arguments: 
- `rng`: Specify a random number generator to use
- `solution_index`: For a vector valued unknown, return the unknown at this index (can be `:` for the full solution)
- `config`: Dict{String,Any} describing optional parameters for the fem_model
"""
function sample_FEM(fem_model::HTTPModel, n_samples::Int; sample_distribution::UnivariateDistribution, rng = default_rng(), solution_index = 1, config = empty_config())::UnivariateFEMSample
    uniform_sample = rand(rng, n_samples)
    fem_sample = draw_FEM_samples(fem_model, quantile(sample_distribution, uniform_sample), solution_index = solution_index, config = config)
    pars = params(sample_distribution)
    DT = pars isa Tuple ? eltype(pars) : valtype(pars)
    return UnivariateFEMSample{DT}(uniform_sample, sample_distribution, fem_sample)
end


"""
    compute_statistics(sample::UnivariateFEMSample{T}) where {T}

Compute empirical mean and standard deviation for the given sample
"""
function compute_statistics(samples::UnivariateFEMSample{T}) where {T}
    μ_sample = similar(samples.fem_sample[1, :])
    σ_sample = similar(samples.fem_sample[1, :])
    local_sample = similar(samples.fem_sample[:, 1])
    for i in eachindex(μ_sample)
        local_sample = samples.fem_sample[:, i]
        μ_sample[i] = mean(local_sample)
        σ_sample[i] = std(local_sample)
    end
    return μ_sample, σ_sample
end
end
