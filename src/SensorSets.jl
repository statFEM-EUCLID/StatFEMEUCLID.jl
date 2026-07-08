# Copyright (c) 2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier: MIT

"""
   SensorSets

This submodule contains structs describing a collection of measurement sensors.
So far, only an `EqualSensorSet` is implemented, 
where all sensors have the same measurement accuracy.
"""
module SensorSets
import Base.size
using Statistics
import Statistics: cov

using UMBridge: HTTPModel
using LinearAlgebra: diagm
using ..FEMClient

export AbstractSensorSet
export EqualSensorSet
export calculate_projection_matrix

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
    size(sensor_set::EqualSensorSet{T}) where T

Returns the size of the `sensor_set`, i.e. the number of sensors.
"""
function Base.size(sensor_set::EqualSensorSet{T}) where {T}
    return length(sensor_set.locations)
end

"""
    cov(sensor_set::EqualSensorSet{T})::T where T

Returns the cov matrix of the `sensor_set`.
For equal sensors this is a diagonal matrix containing the common sensor variance.
"""
function Statistics.cov(sensor_set::EqualSensorSet{T})::AbstractMatrix{T} where {T}
    return (sensor_set.σ)^2 * diagm(ones(T, size(sensor_set)))
end


end
