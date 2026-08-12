# Copyright (c) 2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier: MIT

"""
   StatFEM

This submodule contains functions for performing the statistical FEM on given measurement data and stochastic prior.
The method corrects a given PCE based prior ``u_f`` based on given measurements with independent measurement noise using a Gauß-Markov Kalman Filter.  
"""
module StatFEM

using LinearAlgebra
using PositiveFactorizations: cholesky, Positive
using UMBridge
using ..FEMClient
using ..PCE
using ..SensorSets: AbstractSensorSet, projection_matrix, cov

export posterior_mean, posterior_statistics

"""
    calculate_Kalman_gain(H,C_u,C_e)

Internal, calculates the Kalman gain needed at multiple points.
"""
function calculate_Kalman_gain(H, C_u, C_e)
    S = Symmetric(H * C_u * transpose(H) + C_e)
    S_fact = cholesky(Positive, Matrix(S))
    K = C_u * transpose(H) * inv(S_fact)
    return K
end

"""
    calculate_posterior_mean(mean_measurements,H,K,μ_PCE)

Actually calculate the posterior mean based on `mean_measurements`,
projection matrix `H`, Kalman gain `K` and mean prior displacement.
"""
function calculate_posterior_mean(mean_measurements, H, K, μ_PCE)
    innovation = mean_measurements - H * μ_PCE
    μ_innov = μ_PCE + K * innovation
    return μ_innov
end

"""
    calculate_posterior_covariance(
        H::AbstractMatrix{T}, K::AbstractMatrix{T}, C_u::AbstractMatrix{T}, C_e::AbstractMatrix{T}
    ) where {T}

Actually calculate the posterior covariance matrix based on the projection matrix `H`, Kalman gain `K`,
prior covariance `C_u` and sensor covariance `C_e`
"""
function calculate_posterior_covariance(
        H::AbstractMatrix{T}, K::AbstractMatrix{T}, C_u::AbstractMatrix{T}, C_e::AbstractMatrix{T}
    ) where {T}
    n = size(C_u)[1]
    Id = Matrix{T}(I, n, n)
    A = (Id - K * H)
    C_post = A * C_u * transpose(A) + K * C_e * transpose(K)
    return C_post
end

"""
    corrected_pce_covariance(pce)

Correct the pce covariance so it becomes a symmetric matrix.
"""
function corrected_pce_covariance(pce)
    C_u_raw = PCE.cov(pce)
    C_u = Symmetric((C_u_raw + C_u_raw') ./ 2)
    return C_u
end

"""
    posterior_mean(sensor_set::AbstractSensorSet{T}, measurements::AbstractMatrix{T}, pce) where {T}

Calculate the posterior mean of a PCE based prior given a `sensor_set` and `measurements`.
"""
function posterior_mean(sensor_set::AbstractSensorSet{T}, measurements::AbstractMatrix{T}, pce) where {T}
    n_repetitions = size(measurements)[2]
    mean_measurements = vec(sum(measurements; dims = 2)) / n_repetitions

    C_e = cov(sensor_set) ./ n_repetitions
    H = projection_matrix(sensor_set)
    C_u = corrected_pce_covariance(pce)

    K = calculate_Kalman_gain(H, C_u, C_e)
    posterior_mean = calculate_posterior_mean(mean_measurements, H, K, mean(pce))
    return posterior_mean
end

"""
    posterior_mean(sensor_set::AbstractSensorSet{T}, measurements::AbstractVector{T}, pce) where {T}

Calculate the posterior mean of a PCE based prior given a `sensor_set` and a single measurement.
"""
function posterior_mean(sensor_set::AbstractSensorSet{T}, measurements::AbstractVector{T}, pce) where {T}
    return posterior_mean(sensor_set, reshape(measurements, (length(measurements), 1)), pce)
end

"""
    posterior_statistics(sensor_set::AbstractSensorSet{T},measurements::AbstractArray{T},pce) where {T}

Calculate the posterior statistics (mean & covariance) of a PCE based prior, given a `sensor_set` and `measurements`.
"""
function posterior_statistics(sensor_set::AbstractSensorSet{T}, measurements::AbstractArray{T}, pce) where {T}
    n_repetitions = size(measurements)[2]
    mean_measurements = vec(sum(measurements; dims = 2)) / n_repetitions

    C_e = cov(sensor_set) ./ n_repetitions
    H = projection_matrix(sensor_set)
    C_u = corrected_pce_covariance(pce)

    K = calculate_Kalman_gain(H, C_u, C_e)
    posterior_mean = calculate_posterior_mean(mean_measurements, H, K, mean(pce))
    posterior_covariance = calculate_posterior_covariance(H, K, C_u, C_e)
    return posterior_mean, posterior_covariance
end

"""
    posterior_statistics(sensor_set::AbstractSensorSet{T},measurements::AbstractVector{T},pce) where {T}

Calculate the posterior statistics (mean & covariance) of a PCE based prior, given a `sensor_set` and a single `measurement` vector.
"""
function posterior_statistics(sensor_set::AbstractSensorSet{T}, measurements::AbstractVector{T}, pce) where {T}
    return posterior_statistics(sensor_set, reshape(measurements, (length(measurements), 1)), pce)
end


end
