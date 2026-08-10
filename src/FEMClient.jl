# Copyright (c) 2025-2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier: MIT


"""
    FEMClient

This submodule contains wrapper functions around `UMBridge.jl` for conversion 
between Julia types and what UMBridge.jl expects or returns.
"""
module FEMClient


using UMBridge: evaluate, gradient, HTTPModel
using Mocking: @mock

export evaluate_fem_model

"""
    empty_config()::Dict{String, Any}

Return an empty config for models that don't depend on additional parameters
"""
function empty_config()::Dict{String, Any}
    return Dict{String, Any}()
end


"""
    flatten_if_needed(eval_result)

Older UMBridge versions returned a single vector as a vector of vectors.
If that is the case return the contained vector
"""
function flatten_if_needed(eval_result)
    if typeof(eval_result[1][1]) == Vector{Any}
        return eval_result[1]
    end
    return eval_result
end

"""
    evaluate_fem_model(fem_model, parameter::Float64, solution_index = 1)

Query the server to solve the FEM model `fem_model` for a given (material) parameter.
Keyword arguments:
- `solution_index`: For a vector valued unknown, return the unknown at this index (can be `:` for the full solution)
- `config`: Dict{String,Any} describing optional parameters for the fem_model
"""
function evaluate_fem_model(fem_model::HTTPModel, parameter::Float64; solution_index = 1, config::Dict{String, Any} = empty_config(), extra_params::Vector{Float64} = Float64[])
    return evaluate_fem_model(fem_model, [parameter], solution_index = solution_index, config = config, extra_params = extra_params)
end


"""
    evaluate_fem_model(fem_model::HTTPModel, parameters::Vector{Float64}; solution_index = 1, config::Dict{String,Any} = empty_config())

Query the server to solve the FEM model `fem_model` for a given set of (material) parameters.
Keyword arguments:
- `solution_index`: For a vector valued unknown, return the unknown at this index (can be `:` for the full solution)
- `config`: Dict{String,Any} describing optional parameters for the fem_model
"""
function evaluate_fem_model(fem_model::HTTPModel, parameters::Vector{Float64}; solution_index = 1, config::Dict{String, Any} = empty_config(), extra_params::Vector{Float64} = Float64[])
    full_parameters = [parameters]
    if !isempty(extra_params)
        full_parameters = [parameters, extra_params]
    end
    eval_result = flatten_if_needed(@mock(evaluate(fem_model, full_parameters, config)))

    if solution_index == Colon()
        return Float64.(reduce(vcat, eval_result))
    end
    return Float64.(eval_result[solution_index])
end

"""
    evaluate_projection(projection_model::HTTPModel, sensor_points::Vector{Vector{T}}, n_dofs::Int64) where {T <: Real}

TBW
"""
function evaluate_projection(projection_model::HTTPModel, sensor_points::Vector{Vector{T}}, n_dofs::Int64) where {T <: Real}
    result = zeros(length(sensor_points), n_dofs)
    for i in eachindex(sensor_points)
        request_result = flatten_if_needed(evaluate(projection_model, [sensor_points[i]]))
        global_cell_ids = Int64.(request_result[1])
        cell_basisvalues = Float64.(request_result[2])
        result[i, global_cell_ids] = cell_basisvalues
    end
    return result
end

"""
    evaluate_fem_residual(residual_model::HTTPModel,u::Vector{T},κ::Vector{T};config::Dict{String,Any}=empty_config()) where {T<: AbstractFloat}

TBW
"""
function evaluate_fem_residual(residual_model::HTTPModel, u::Vector{T}, κ::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}
    return evaluate(residual_model, [u, κ], config)[1][1]
end

"""
    evaluate_fem_residual_gradient(residual_model::HTTPModel,u::Vector{T},κ::Vector{T};config::Dict{String,Any}=empty_config()) where {T<:AbstractFloat}

TBW
"""
function evaluate_fem_residual_gradient(residual_model::HTTPModel, u::Vector{T}, κ::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}
    return gradient(
        residual_model,
        0,
        1,
        [u, κ],
        [1.0],
        config
    )
end

end #module
