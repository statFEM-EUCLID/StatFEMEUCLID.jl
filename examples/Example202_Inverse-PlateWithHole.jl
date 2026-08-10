# Copyright (c) 2026 Jan Philipp Thiele
# SPDX-License-Identifier MIT

#=

# 202: Model parameter identification for a 2D Plate with a hole
([source code](@__SOURCE_URL__))

In this example different methods for identification of constitutive model parameters are used 
to re-identify a known truth that was used to generate synthetic data.

The underlying setup is the same as in the previous example.
For the synthetic data the main difference is that we now save the noiseless full-field information, i.e. 
at all nodes of the mesh.

For the parameter identification we differentiate between two options
- calibration, where we assume the model is known and use a goal function similar to the virtual fields and equilibrium gap methods to identify the parameters
- discovery, where we don't assume a specific model, but use a larger model library and additional regularization to select the relevant parameters

## Results

For both approaches we compare two different optimizers, a gradient-free Nelder-Mead (NM) and a gradient-based BFGS.
The gradient-free optimizer is especially interesting if your forward model can not provide the gradient of the residual w.r.t to the material parameters,
as numeric differentiation is not necessarily robust.

### Calibration

For the pure calibration setup, where the material model is known but its parameters $\kappa_{\text{true}}$ are assumed to be unknown,
the goal functional is the L2-norm of the FEM residual vector. 
Both optimizations take a similar amount of time for this setup and the identified parameters are shown in the table below. 

|          | $\kappa_{\text{true}}$ | $\kappa_{\text{NM}}^*$ | $\kappa_{\text{BFGS}}^*$ |
|:--------:| ---------------------- | ---------------------- | ------------------------ |
| $A_{10}$ |                    0.3 |     0.2999996355675296 |       0.2999996518345033 |
| $A_{01}$ |                    0.2 |     0.2000003849699102 |       0.2000003664689581 |
| $B_{1}$  |                    1.5 |     1.5000001106084495 |       1.5000001073678113 | 

### Discovery

For the discovery the goal functional also has an L1-regularization term.
Additionally, for this specific setup with a large Mooney-Rivlin model,
we add a penalization term to ensure compressibility.
Since this is a harder optimization problem we also compare results with an outer primal interior point method (IP) around NM or BFGS.
On a Laptop with an Intel Core Ultra 7 265U processor the fastest method was BFGS taking around 60 seconds,
so be aware of this when running the example.
All other EUCLID runs take significantly longer (NM: 580s, IP-NM: 1280s, IP-BFGS:240s), 
so they are commented out by default.

|          | $\kappa_{\text{true}}$ | $\kappa_{\text{NM}}^*$ | $\kappa_{\text{BFGS}}^*$ | $\kappa_{\text{IP-NM}}^*$ | $\kappa_{\text{IP-BFGS}}^*$ |
|:--------:| ---------------------- |:---------------------- |:------------------------ |:------------------------- |:--------------------------- |
| $A_{10}$ |                    0.3 | 0.30217189437873715    | 0.3000002740255784       | 0.2845441199595386        | 0.2999997391289652          |
| $A_{01}$ |                    0.2 | 0.19280542741216714    | 0.1999997277608331       | 0.2161550941421935        | 0.2000002734711289          |
| $A_{20}$ |                    0.0 | 0.0                    | 0.0000110892436136       | 0.0000017833983007        | 0.0                         |
| $A_{11}$ |                    0.0 | 0.05506791508307378    | 0.0                      | 0.0001403260875725        | 0.0                         |
| $A_{02}$ |                    0.0 | 0.0                    | 0.0000148435620673       | 0.0000001763018293        | 0.0                         |
| $A_{30}$ |                    0.0 | 0.0                    | 0.0                      | 0.0001085739641259        | 0.0000000150109834          |
| $A_{21}$ |                    0.0 | 0.01578347281041807    | 0.0000181101673901       | 0.0001589385506675        | 0.0                         |
| $A_{12}$ |                    0.0 | 0.0                    | 0.0000017408432304       | 0.0000020167020899        | 0.0                         |
| $A_{03}$ |                    0.0 | 0.05380845291756604    | 0.0                      | 0.0003512676660859        | 0.0                         |
| $B_{1}$  |                    1.5 | 1.4995740646402762     | 1.4999999756971945       | 1.5043888013198317        | 1.5000000828350004          |


!!! warning "Important"

    Remember to start your UMBridge FEM server before running the example!


=#

module Example202_EUCLID_PlateWithHole

using StatFEMEUCLID
using Random
using Distributions
using CairoMakie
using Gmsh
using ExtendableGrids
using GridVisualize
using CSV, DataFrames
using Optim
using UMBridge
using LineSearches
using LinearAlgebra

# With `server_url` you can point to a different UMBridge server implementing the model
# the default is what is provided by the Ferrite.jl-based implementation [here](https://github.com/statFEM-EUCLID/HolePlate2D_Ferrite.jl/)

function main(;
        server_url = "http://localhost:4232",
        F = 0.5
    )

    model_name = "HolePlate2D"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)
    residual_model = UMBridge.HTTPModel(model_name * ".residual", server_url)

    config_cal = Dict{String, Any}(
        "material" => "MooneyRivlin",
        "n_MR" => 1,
        "n_VOL" => 1
    )

    κ_0_cal = [0.12, 0.12, 0.72]

    dataframe_measurments = CSV.read("Example202_measurements.csv", DataFrame)
    measurements = [Vector{Float64}(row) for row in eachrow(dataframe_measurments)]

    n_dofs = length(measurements)
    u_measurements = zeros(2 * n_dofs)

    for i in eachindex(measurements)
        u_measurements[i] = measurements[i][1]
        u_measurements[n_dofs + i] = measurements[i][2]
    end


    f_cal, g_cal! = create_residual_objective_and_gradient(residual_model, u_measurements; config = config_cal)

    optimal_residual_cal_NM = optimize(f_cal, κ_0_cal, NelderMead())

    @show optimal_residual_cal_NM
    κ_opt_cal_NM = Optim.minimizer(optimal_residual_cal_NM)
    @show κ_opt_cal_NM

    optimal_residual_cal_BFGS = optimize(
        f_cal, g_cal!, κ_0_cal,
        BFGS(linesearch = LineSearches.BackTracking(order = 3))
    )

    @show optimal_residual_cal_BFGS
    κ_opt_cal_BFGS = Optim.minimizer(optimal_residual_cal_BFGS)
    @show κ_opt_cal_BFGS

    config_disc = Dict{String, Any}(
        "material" => "MooneyRivlin",
        "n_MR" => 3,
        "n_VOL" => 1
    )

    # Since we have to pass this information to multiple functions,
    # we combine the two UMBridge models with their config and the mean traction.
    euclid_config = InverseModelConfig(fem_model, residual_model, config_disc, F)

    κ_0_disc = [0.4, 0.4, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 6.3]

    lasso_sweep = 5 * logrange(1.0e-5, 1.0, 6)


    f_disc, g_disc! = create_residual_objective_and_gradient(residual_model, reshape(u_measurements, n_dofs * 2); config = config_disc)
    f_mr_penalty, g_mr_penalty! = MooneyRivlin_penalization(3.0, 10.0)
    function f_penalized(κ)
        return f_disc(κ) + f_mr_penalty(κ)
    end
    function g_penalized!(G, κ)
        g_disc!(G, κ)
        return g_mr_penalty!(G, κ) #written to add on to G
    end

    sensor_set = FullFieldSensorSet(n_dofs)


    @info "EUCLID with BFGS"
    EUCLID(euclid_config, lasso_sweep, κ_0_disc, sensor_set, u_measurements, f_penalty = f_mr_penalty, g_penalty = g_mr_penalty!, show_info = true, use_gradient = true)

    #= The following EUCLID calls are all commented out because they take significantly longer.

    @info "EUCLID with NelderMead"
    EUCLID(euclid_config,lasso_sweep,κ_0_disc,sensor_set,u_measurements,
    f_penalty=f_mr_penalty,g_penalty=g_mr_penalty!,
    show_info=true,use_gradient=false,
    optimizer=NelderMead(),
    optimizer_options=Optim.Options(g_abstol=1.0e-10,iterations=10_000))

    lower = zeros(length(κ_0_disc))
    upper = 50.0.*ones(length(κ_0_disc))


    @info "EUCLID with Fminbox NelderMead"
    EUCLID(euclid_config,lasso_sweep,κ_0_disc,sensor_set,u_measurements,
    f_penalty=f_mr_penalty,g_penalty=g_mr_penalty!,
    upper=upper,lower=lower,
    show_info=true,use_gradient=false,
    optimizer=Fminbox(NelderMead()),
    optimizer_options=Optim.Options(time_limit=200))

    @info "EUCLID with Fminbox BFGS"
    EUCLID(euclid_config,lasso_sweep,κ_0_disc,sensor_set,u_measurements,
        f_penalty=f_mr_penalty,g_penalty=g_mr_penalty!,
        upper=upper,lower=lower,
        show_info=true,use_gradient=true,
        optimizer=Fminbox(BFGS(linesearch=LineSearches.BackTracking())),
        optimizer_options=Optim.Options(time_limit=200)
    )

    =#
    return nothing
end

function MooneyRivlin_penalization(penalty_multiplier, penalization_weight)
    function f(κ)
        return abs(sum(κ[1:(end - 1)]) * penalty_multiplier - κ[end]) * penalization_weight
    end
    function g!(G, κ)
        s = sign(sum(κ[1:(end - 1)] * penalty_multiplier) - κ[end])
        G[1:(end - 1)] .+= penalization_weight * penalty_multiplier * s
        return G[end] -= penalization_weight * s
    end

    return f, g!
end


# This function generates the synthetic data based on a Mooney-Rivlin model.
# Note that the forward model has to be able to solve with this material model to
# regenerate the data.
function generate_synthetic_data(;
        F = 0.5,
        server_url = "http://localhost:4232"
    )

    # In contrast to the previous example, we are using full field data,
    # so we don't need the projection here.
    model_name = "HolePlate2D"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)

    A_10 = 0.3
    A_01 = 0.2
    B_1 = 1.5

    κ = [A_10, A_01, B_1]

    # We want to solve with the smallest Mooney-Rivlin model
    # with a total of three parameters
    MR_material_config = Dict{String, Any}(
        "material" => "MooneyRivlin",
        "n_MR" => 1,
        "n_VOL" => 1
    )
    n_dofs = UMBridge.model_output_sizes(fem_model)[1]


    # Since we solve with the exact mean traction force,
    # we don't have to sample and can directly evaluate the
    # forward model once.
    u_truth = zeros(n_dofs, 2)
    u_truth_full = evaluate_fem_model(
        fem_model, F, solution_index = :, config = MR_material_config, extra_params = κ
    )
    u_truth[:, 1] = u_truth_full[1:n_dofs]
    u_truth[:, 2] = u_truth_full[(n_dofs + 1):end]

    # and finally write the result to file
    dataframe_measurements = DataFrame(u_x = u_truth[:, 1], u_y = u_truth[:, 2])
    return CSV.write("Example202_measurements.csv", dataframe_measurements)
end

end
