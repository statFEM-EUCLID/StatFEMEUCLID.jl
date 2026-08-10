# Copyright (c) 2025-2026 Jan Philipp Thiele
# SPDX-License-Identifier: MIT

"""
    Inverse

This submodule contains functions for solving inverse problems 
to identify material parameters.
"""
module Inverse

using ..FEMClient: evaluate_fem_model, evaluate_fem_residual, evaluate_fem_residual_gradient
using ..SensorSets: project, AbstractSensorSet
using UMBridge: HTTPModel, supports_gradient
using LinearAlgebra: norm
using Optim, LineSearches

export create_residual_objective, create_residual_objective_and_gradient
export AbstractModelSelectionCriterion, AikaikeInformationCriterion, AIC, BayesianInformationCriterion, BIC, evaluate_criterion
export InverseModelConfig, EUCLID

"""
    create_residual_objective(residual_model::HTTPModel, fem_solution::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}

Returns an optimization objective w.r.t. the material parameters of a model.
"""
function create_residual_objective(residual_model::HTTPModel, fem_solution::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}
    return function f(x)
        return evaluate_fem_residual(residual_model, fem_solution, x; config = config)
    end
end

"""
    create_residual_objective_and_gradient(residual_model::HTTPModel, fem_solution::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}

Returns an optimization objective w.r.t. the material parameters κ of a model 
as well as its derivative wr.t. κ
"""
function create_residual_objective_and_gradient(residual_model::HTTPModel, fem_solution::Vector{T}; config::Dict{String, Any} = empty_config()) where {T <: AbstractFloat}
    f = create_residual_objective(residual_model, fem_solution; config = config)

    #TODO: check if model provides gradient, otherwise error
    if !supports_gradient(residual_model)
        ArgumentError("")
    end

    function g!(G, x)
        return G .= evaluate_fem_residual_gradient(residual_model, fem_solution, x; config = config)
    end
    return f, g!
end

abstract type AbstractModelSelectionCriterion end

"""
    struct AikaikeInformationCriterion

One choice of information criterion for model selection (short form: AIC)
See the implementation of `evaluate_criterion` for the formula
"""
struct AikaikeInformationCriterion <: AbstractModelSelectionCriterion end
"""
    struct BayesianInformationCriterion

One choice of information criterion for model selection (short form: BIC)
See the implementation of `evaluate_criterion` for the formula
"""
struct BayesianInformationCriterion <: AbstractModelSelectionCriterion end

const AIC = AikaikeInformationCriterion
const BIC = BayesianInformationCriterion

"""
    evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, ::Type{AbstractModelSelectionCriterion})

Evaluate the model selection criterion 
"""
function evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, ::Type{AbstractModelSelectionCriterion})
    return @error "Unknown model selection criterion!"
end

"""
    evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, ::Type{AikaikeInformationCriterion})

Evaluate the model selection criterion based on AIC
"""
function evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, ::Type{AikaikeInformationCriterion})
    return n_sensor * log(rmse) + 2 * n_κ
end

"""
    evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, criterion::Type{BayesianInformationCriterion})

Evaluate the model selection criterion based on BIC
"""
function evaluate_criterion(n_κ::Int64, n_sensor::Int64, rmse::Float64, ::Type{BayesianInformationCriterion})
    return n_sensor * log(rmse) + n_κ * log(n_sensor)
end

"""
    model_rmse(fem_model::HTTPModel,μ_T::T,server_config::Dict{String,Any},κ::Vector{T},sensor_set,measurements::Vector{T}) where {T}

Calculate the RMSE (root mean squared error) between an FEM solution for given set of material parameters κ
and a given set of measurements.
"""
function model_rmse(fem_model::HTTPModel, μ_T::T, server_config::Dict{String, Any}, κ::Vector{T}, sensor_set, measurements::Vector{T}) where {T}
    u_κ = evaluate_fem_model(fem_model, μ_T, solution_index = :, config = server_config, extra_params = κ)
    return model_rmse(u_κ, sensor_set, measurements)
end

"""
    model_rmse(u_κ,sensor_set,measurements)

Calculate the RMSE (root mean squared error) between a given FEM solution u and a set of measurements.
"""
function model_rmse(u_κ, sensor_set, measurements)
    Hu_k = project(sensor_set, u_κ)
    return norm(Hu_k - measurements, 2) / sqrt(size(sensor_set))
end

"""
    create_lasso_objective_and_gradient(λ::T,f::Function,g!::Function,f_penalty::Function,g_penalty!) where T

Internal: Add LASSO regularization to a given `f` as well as its gradient `g!`.
Optionally, add a penalization function `f_penalty` and its matching gradient `g_penalty!`
"""
function create_lasso_objective_and_gradient(λ::T, f::Function, g!::Function, f_penalty::Function, g_penalty!) where {T}
    function f_lasso(κ)
        return f(κ) + f_penalty(κ) + λ * norm(κ, 1)
    end
    function g_lasso!(G, κ)
        g!(G, κ)
        g_penalty!(G, κ)
        return G += λ * sign.(κ)
    end
    return f_lasso, g_lasso!
end

"""
    struct InverseModelConfig{T}

Common combination of arguments needed for inverse models,
so they are combined in a struct.

# Fields
- `fem_model::HTTPModel` - model providing the FEM evaluation (F,κ)->u
- `residual_model::HTTPModel` - model providing the FEM residual
- `config::Dict{String,Any}` - model config 
- `μ::T` - mean traction
"""
struct InverseModelConfig{T}
    fem_model::HTTPModel
    residual_model::HTTPModel
    config::Dict{String, Any}
    μ::T
end

"""
    EUCLID(
    model_config::InverseModelConfig{T},
    lasso_sweep::Vector{T},
    κ_0::Vector{T},
    sensor_set::AbstractSensorSet{T},
    measurements::Vector{T}
    ;
    optimizer::Optim.AbstractOptimizer=BFGS(linesearch = LineSearches.BackTracking()),
    optimizer_options::Optim.Options = Optim.Options(),
    upper=NaN,
    lower=NaN,
    threshold::T = 1.0e-9,
    selection_criterion::Type{C} = BIC,
    f_penalty::Function=κ::Vector{T}->0,
    g_penalty::Function=(G,κ)->nothing,
    show_info::Bool=false,
    use_gradient=false
    ) where {T <: Real, C <: AbstractModelSelectionCriterion}

Perform the Efficient Unsupervised Constitutive Law identification & Discovery (EUCLID).
Internally, this performs a LASSO regularization along the given `lasso_sweep`.
"""
function EUCLID(
        model_config::InverseModelConfig{T},
        lasso_sweep::Vector{T},
        κ_0::Vector{T},
        sensor_set::AbstractSensorSet{T},
        measurements::Vector{T}
        ;
        optimizer::Optim.AbstractOptimizer = BFGS(linesearch = LineSearches.BackTracking()),
        optimizer_options::Optim.Options = Optim.Options(),
        upper = NaN,
        lower = NaN,
        threshold::T = 1.0e-9,
        selection_criterion::Type{C} = BIC,
        f_penalty::Function = κ::Vector{T} -> 0,
        g_penalty::Function = (G, κ) -> nothing,
        show_info::Bool = false,
        use_gradient = false
    ) where {T <: Real, C <: AbstractModelSelectionCriterion}

    κ_opt = similar(κ_0)
    λ_opt = typemax(T)
    crit_opt = typemax(T)

    f_resid, g_resid! = if (use_gradient)
        create_residual_objective_and_gradient(model_config.residual_model, measurements; config = model_config.config)
    else
        create_residual_objective(model_config.residual_model, measurements; config = model_config.config), g_penalty
    end
    for λ in lasso_sweep
        f_lasso, g_lasso! = create_lasso_objective_and_gradient(λ, f_resid, g_resid!, f_penalty, g_penalty)
        optimum =
        if (sum(isnan.(lower) .+ isnan.(upper)) == 0) #use constrained optimization
            if (use_gradient)
                optimize(f_lasso, g_lasso!, lower, upper, κ_0, optimizer, optimizer_options)
            else
                optimize(f_lasso, lower, upper, κ_0, optimizer, optimizer_options)
            end
        else
            if (use_gradient)
                optimize(f_lasso, g_lasso!, κ_0, optimizer, optimizer_options)
            else
                optimize(f_lasso, κ_0, optimizer, optimizer_options)
            end
        end
        κ_λ = Optim.minimizer(optimum)
        κ_λ[κ_λ .< threshold] .= zero(T)
        rmse = model_rmse(model_config.fem_model, model_config.μ, model_config.config, κ_λ, sensor_set, measurements)
        crit = evaluate_criterion(sum(κ_λ .> 0.0), size(sensor_set), rmse, selection_criterion)
        if crit < crit_opt
            crit_opt = crit
            κ_opt = κ_λ
            λ_opt = λ
        end
        if show_info
            @info "Values for λ=$λ" κ_λ, rmse, crit, Optim.minimum(optimum)
        end
    end
    if show_info
        @info "Discovered model with λ=$λ_opt" κ_opt
    end
    return κ_opt, λ_opt
end
end
