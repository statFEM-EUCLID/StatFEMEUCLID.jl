# Copyright (c) 2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier: MIT

"""
   StatFEM

This submodule contains functions for 
"""
module StatFEM

using LinearAlgebra: I, Symmetric, inv
using PositiveFactorizations: cholesky, Positive
using UMBridge
using ..FEMClient
using ..PCE

export SensorSet
export EqualSensorSet
export size

abstract type SensorSet{T} end

mutable struct EqualSensorSet{T} <: SensorSet{T}
    const locations::Vector{Vector{T}}
    const σ::T
    projection::Matrix{T} 
end


function EqualSensorSet(locations::Vector{Vector{T}},σ::T) where T
    return EqualSensorSet(locations,σ,Array{T}(undef,0,0))
end


function get_projection_matrix(sensor_set::EqualSensorSet{T},projection_model::UMBridge.HTTPModel,n_dofs::Int64) where T
    sensor_set.projection = FEMClient.evaluate_projection(projection_model,sensor_set.locations,n_dofs)
end

"""
    size(sensor_set::EqualSensorSet{T}) where T

TBW
"""
function size(sensor_set::EqualSensorSet{T}) where T
    return length(sensor_set.locations)
end

"""
    covariance(sensor_set::EqualSensorSet{T})::T where T

TBW
"""
function covariance(sensor_set::EqualSensorSet{T})::AbstractMatrix{T} where T
    return Matrix((sensor_set.σ ^ 2)*I(size(sensor_set)))
end

"""
    calculate_Kalman_gain(H,C_u,C_e)

TBW
"""
function calculate_Kalman_gain(H,C_u,C_e)
    S = Symmetric(H * C_u * transpose(H) + C_e)
    S_fact = cholesky(Positive,Matrix(S))
    K = C_u * transpose(H) * inv(S_fact)
    return K
end

"""
    calculate_posterior_mean(mean_measurements,H,K,μ_PC)

TBW
"""
function calculate_posterior_mean(mean_measurements,H,K,μ_PC)
    innovation = mean_measurements - H * μ_PC
    μ_innov = μ_PC + K * innovation
    return μ_innov
end

"""
    corrected_pce_covariance(pce)

TBW
"""
function corrected_pce_covariance(pce)
    C_u_raw = PCE.covariance(pce)
    C_u = Symmetric((C_u_raw+C_u_raw')./2)
    return C_u
end

"""
    posterior_mean(sensor_set::SensorSet{T},measurements,pce)

TBW
"""
function posterior_mean(sensor_set::EqualSensorSet{T},measurements,pce) where T
    n_repetitions = ndims(measurements) == 1 ? 1 : size(measurements,2)
    C_e = covariance(sensor_set)./n_repetitions
    H = sensor_set.projection
    C_u = corrected_pce_covariance(pce)
    
    #mean measurements
    mean_measurements = ndims(measurements) == 1 ? vec(measurements) : vec(sum(measurements;dims=2))/n_repetitions

    K = calculate_Kalman_gain(H,C_u,C_e)
    posterior_mean = calculate_posterior_mean(mean_measurements,H,K,mean(pce))
    return posterior_mean
end

end
