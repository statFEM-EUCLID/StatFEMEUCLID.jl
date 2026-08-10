# Copyright (c) 2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier: MIT

"""
   SensorSets

This submodule contains structs describing a collection of measurement sensors.
"""
module SensorSets
import Base.size
using Random: AbstractRNG, default_rng
using Distributions: Normal
using Statistics
import Statistics: cov

using UMBridge: HTTPModel
using LinearAlgebra: diagm
using ..FEMClient

using CSV, DataFrames
import CSV: read

export AbstractSensorSet
export EqualSensorSet, FullFieldSensorSet
export calculate_projection_matrix
export project, projection_matrix, apply_noise!

"""
    abstract type AbstractSensorSet{T}


"""
abstract type AbstractSensorSet{T} end

"""
    mutable struct EqualSensorSet{T} <: AbstractSensorSet{T}

A set of sensors with equal measurement accuracies given by a common standard deviation.

# Fields
- `const locations::Vector{Vector{T}}` - vector of sensor locations (as coordinate vectors)
- `const σ::T` - common standard deviation
- `projection::Matrix{T}` - Projection matrix from degrees of freedom of an FEM solution to the sensor locations
"""
mutable struct EqualSensorSet{T} <: AbstractSensorSet{T}
    const locations::Vector{Vector{T}}
    const σ::T
    projection::Matrix{T}
end


"""
    EqualSensorSet(locations::Vector{Vector{T}}, σ::T) where {T<:AbstractFloat}

Construct an EqualSensorSet through given measurement locations and a common standard deviation 
for all sensors.
Note: For use within the statistical FEM, `calculate_projection_matrix` has to be called as well.
"""
function EqualSensorSet(locations::Vector{Vector{T}}, σ::T) where {T <: AbstractFloat}
    return EqualSensorSet(locations, σ, Array{T}(undef, 0, 0))
end


"""
    struct FullFieldSensorSet{T} <: AbstractSensorSet{T}

A mock sensor set when the full FE field without noise is measured.
For example when using EUCLID without StatFEM.

# Fields
- `n_dofs::Int{T}` - number of unknowns in one dimension
- `projection::Matrix{T}` - Projection matrix from degrees of freedom of an FEM solution to themselves (identity)
"""
struct FullFieldSensorSet{T} <: AbstractSensorSet{T}
    n_dofs::Int64
    projection::Matrix{T}
end

"""
    FullFieldSensorSet(::Type{T}, n_dofs::Int64) where {T <: AbstractFloat}

Construct a FullFieldSensorSet with matrix value type T.
"""
function FullFieldSensorSet(::Type{T}, n_dofs::Int64) where {T <: AbstractFloat}
    return FullFieldSensorSet(n_dofs, diagm(ones(T, n_dofs)))
end

"""
    FullFieldSensorSet(n_dofs::Int64)

Construct a FullFieldSensorSet with default matrix value type Float64.
"""
function FullFieldSensorSet(n_dofs::Int64)
    return FullFieldSensorSet(Float64, n_dofs)
end


"""
    calculate_projection_matrix(sensor_set::EqualSensorSet{T}, projection_model::HTTPModel, n_dofs::Int64) where {T}

Query the `projection_model` to calculate a projection matrix based on the sensor locations of the `sensor_set`.
This matrix maps from the FEM degrees of freedom to the sensor locations
"""
function calculate_projection_matrix(sensor_set::EqualSensorSet{T}, projection_model::HTTPModel, n_dofs::Int64) where {T}
    return sensor_set.projection = FEMClient.evaluate_projection(projection_model, sensor_set.locations, n_dofs)
end


"""
    projection_matrix(sensor_set::EqualSensorSet{T}) where T

Returns the projection matrix of the `sensor_set` mapping from FEM degrees of freedom to the sensor locations.
"""
function projection_matrix(sensor_set::EqualSensorSet{T}) where {T}
    return sensor_set.projection
end


"""
    projection_matrix(sensor_set::FullFieldSensorSet{T}) where {T}

Returns the projection matrix of the `sensor_set` mapping from FEM degrees of freedom to the sensor locations.
"""
function projection_matrix(sensor_set::FullFieldSensorSet{T}) where {T}
    return sensor_set.projection
end

"""
    project(sensor_set::AbstractSensorSet{T},full_field_measurement::Matrix{T}) where {T}

Apply the projection matrix of the `sensor_set` to the `full_field_measurement` 
to obtain a measurement at the sensor locations.
"""
function project(sensor_set::AbstractSensorSet{T}, full_field_measurement::Vector{T}) where {T}
    n_dofs = size(projection_matrix(sensor_set))[2]
    dim = div(length(full_field_measurement), n_dofs)
    n_sen = size(sensor_set)
    projected_measurements = zeros(T, n_sen * dim)
    for i in 0:(dim - 1)
        projected_measurements[(1:n_sen) .+ (i * n_sen)] =
            projection_matrix(sensor_set) * full_field_measurement[(1:n_dofs) .+ (i * n_dofs)]
    end
    return projected_measurements
end

"""
    size(sensor_set::EqualSensorSet{T}) where T

Returns the size of the `sensor_set`, i.e. the number of sensors.
"""
function Base.size(sensor_set::EqualSensorSet{T}) where {T}
    return length(sensor_set.locations)
end

"""
    size(sensor_set::FullFieldSensorSet{T}) where T

Returns the size of the `sensor_set`, i.e. the number of nodes/degrees of freedom of a scalar variable.
"""
function Base.size(sensor_set::FullFieldSensorSet{T}) where {T}
    return sensor_set.n_dofs
end

"""
    cov(sensor_set::EqualSensorSet{T})::T where T

Returns the cov matrix of the `sensor_set`.
For equal sensors this is a diagonal matrix containing the common sensor variance.
"""
function Statistics.cov(sensor_set::EqualSensorSet{T})::AbstractMatrix{T} where {T}
    return (sensor_set.σ)^2 * diagm(ones(T, size(sensor_set)))
end

"""
    apply_noise!(sensor_set::EqualSensorSet{T},projected_measurement::Matrix{T};rng::AbstractRNG=default_rng()) where {T}

Sample noise from a Normal distribution with standard deviation `sensor_set.σ` and 
add the noise to the measurement at sensor locations.
"""
function apply_noise!(sensor_set::EqualSensorSet{T}, projected_measurement::Matrix{T}; rng::AbstractRNG = default_rng()) where {T}
    noise_distribution = Normal(0.0, sensor_set.σ)
    for i in 1:size(projected_measurement)[2]
        noise = rand(rng, noise_distribution, size(sensor_set))
        projected_measurement[:, i] += noise
    end
    return
end

"""
    CSV.read(source,::Type{EqualSensorSet{T}},σ_sensor::T=1.0;kwargs...) where {T <: AbstractFloat}

Read in a CSV of sensor points to create a set of sensors with equal measurement error
"""
function CSV.read(source, ::Type{EqualSensorSet{T}}, σ_sensor::T = 1.0; kwargs...) where {T <: AbstractFloat}
    data = CSV.read(source, DataFrame; kwargs...)
    return EqualSensorSet([Vector{T}(row) for row in eachrow(data)], σ_sensor)
end

end
