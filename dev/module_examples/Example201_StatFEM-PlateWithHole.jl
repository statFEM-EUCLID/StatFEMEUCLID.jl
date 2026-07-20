# Copyright (c) 2026 Jan Philipp Thiele and Andrea Morini
# SPDX-License-Identifier MIT

#=

# 201: Using StatFEM on a 2D Plate with a hole
([source code](@__SOURCE_URL__))


In this example the statistical finite element method (StatFEM) 
is used to calculate a Bayesian posterior of a sampled finite element solution (prior) based on given data.

The underlying setup is a 2-dimensional plate with a hole.
The mesh is given through `Example201_mesh.msh`.
A traction force is applied to the right boundary (boundary id 1)
The left and bottom boundaries have symmetry conditions such that 
$u_x$ and $u_y$ are constrained to 0 respectively.
The mean force is given as $\mu_F=0.5$.

The synthetic measurement data is obtained by solving the problem with a Mooney-Rivlin model
with $\mu_F$. The full field solution is then mapped to a known set of sensor locations and further perturbed
by Gaussian noise on the sensor points. To regenerate the data you need a server that can solve the problem 
with Mooney Rivlin and pass the appropriate `server_url` to the `generate_synthetic_data` function.
The results are saved and given as `Example201_measurements.csv` such that the following prior sampling
and statFEM can be performed with a server capable of solving the problem with linear elasticity.

The prior is obtained by solving a the problem with linear elasticity 
with matching material parameters. Additionally, the prior
is obtained as a PCE sample with a normally distributed traction force with $\sigma_F=2.5e-2$.


!!! warning "Important"

    Remember to start your UMBridge FEM server before running the example!

=#

module Example201_StatFEM_PlateWithHole

using StatFEMEUCLID
using Random
using Distributions
using UMBridge
using CairoMakie
using Gmsh
using ExtendableGrids
using GridVisualize
using CSV, DataFrames
using PolyChaos
using LinearAlgebra: diag
# If the forward model does not provide a projection (see below) we can use the StatFEMEUCLID extension enabled by loading ExtendableFEMBase
using ExtendableFEMBase

# With `server_url` you can point to a different UMBridge server implementing the model
# the default is what is provided by the Ferrite.jl-based implementation [here](https://github.com/statFEM-EUCLID/HolePlate2D_Ferrite.jl/).
# An alternative is provided by the Kratos-based implementation [here](https://github.com/statFEM-EUCLID/HolePlate2D_Kratos.py/)
# with `server_url="http://localhost:4242`.
# Note that the Kratos implementation only provides the linear elasticity model, so it can not be used for the data generation.

function main(;
        μ_F = 0.5,
        σ_F = 2.5e-2,
        n_PCE = 30,
        E = 2.7,
        ν = 0.35,
        σ_sensor = 0.5,
        server_url = "http://localhost:4232",
    )

    rng = MersenneTwister(2530)  #fixed seed for comparability between runs

    # As in Example101 we start by defining UMBridge HTML models for querying the forward model
    model_name = "HolePlate2D"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)
    # Additionally, we need the projection from degrees of freedom to the sensor locations.
    # The ordering should follow the ordering given by the mesh file, otherwise the plotting will produce wrong results.
    # The consistent option has the server provide this mapping.
    projection_model = UMBridge.HTTPModel(model_name * ".projection", server_url)

    # For calculating the projection on the client side we have to provide the grid
    # but we also need it for plotting, so we load it early
    grid = ExtendableGrids.simplexgrid_from_gmsh("Example201_mesh.msh", Tc = Float64, Ti = Int64)

    # Since some of our example servers can solve for more than one material model
    # we now need to specify this in a configuration.
    # However, if the server only solves linear elasticity this can and would be ignored by default.
    LE_material_config = Dict{String, Any}(
        "material" => "LinElastic"
    )

    # In general, we want to be able to change other parameters, in the case the material parameters.
    # Therefore, these are passed as additional arguments through UMBridge, such that the server should expect
    # an input vector `[[F],[\nu,E]]` in case of linear elasticity, or in general `[[F],[extra_params]]`.
    # The additional parameters are passed with the optional `extra_params` keyword argument.
    ux_prior_sample = sample_FEM(
        fem_model,
        n_PCE,
        sample_distribution = Normal(μ_F, σ_F),
        rng = rng,
        config = LE_material_config,
        extra_params = [ν, E]
    )

    # For statFEM we need three components, a PCE surrogate, measurement data and
    # a projection matrix H from our solution space (dofs) to the locations of the
    # sensors where the measurements were taken.

    # 1 PCE surrogate, see Example101
    ux_pce_surrogate = PolyChaosExpansion(ux_prior_sample, polynomials = GaussOrthoPoly(4))

    # 2 Sensor points from file + projection matrix
    dataframe_sensors = CSV.read("Example201_sensors.csv", DataFrame)
    sensor_points = [Vector{Float64}(row) for row in eachrow(dataframe_sensors)]

    sensor_set = EqualSensorSet(sensor_points, σ_sensor)

    # If your forward model provides the projection you build the matrix with the following commands
    # ```
    # n_dofs = UMBridge.model_output_sizes(fem_model)[1]
    # calculate_projection_matrix(sensor_set, projection_model, n_dofs)
    # ```

    # Otherwise the extension provides (needs `using ExtendableFEMBase`)
    calculate_projection_matrix(sensor_set, grid)

    # 3 Read in (synthetic) measurements
    dataframe_measurements = CSV.read("Example201_measurements.csv", DataFrame)
    measurements = [Vector{Float64}(row) for row in eachrow(dataframe_measurements)]

    ux_measurements = zeros(length(measurements))
    for i in eachindex(measurements)
        ux_measurements[i] = measurements[i][1]
    end

    # With all components in place, we can now calculate the posterior statistics,
    # i.e. the vector of nodal mean values and covariance matrix.
    ux_posterior, C_ux_posterior = posterior_statistics(sensor_set, ux_measurements, ux_pce_surrogate)
    σ_ux_posterior = sqrt.(diag(C_ux_posterior))


    # Finally, we want to plot the sensor locations as well as
    # the prior and posterior along the bottom boundary
    # Plot of sensor points and mesh
    sensorvis = plot_sensors(grid, sensor_points)

    displt = plot_displacements_along_bottom_boundary(grid, sensor_points, measurements, ux_pce_surrogate, ux_posterior, σ_ux_posterior)

    return sensorvis, displt
end

# This function plots the measurements as well as prior and posterior displacements with confidence intervals
# on the line along the bottom boundary of the domain.
function plot_displacements_along_bottom_boundary(grid, sensor_points, measurements, pce_surrogate, μ_posterior, σ_posterior)
    f = Figure(size = (800, 800))
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "u_x")

    # Get the subgrid corresponding to the bottom boundary (boundary id=3)
    sub = subgrid(grid, [3], boundary = true)
    bottom_x_grid = Float64.(sub[Coordinates])[1, :]
    # Need to specify full name here as compute_statistics is also part of Statistics
    μ_PCE, σ_PCE = StatFEMEUCLID.PCE.compute_statistics(pce_surrogate)

    # Extract the values at the boundary DoFs
    bottom_μ_PCE = μ_PCE[sub[NodeParents]]
    bottom_σ_PCE = σ_PCE[sub[NodeParents]]
    bottom_μ_post = μ_posterior[sub[NodeParents]]
    bottom_σ_post = σ_posterior[sub[NodeParents]]

    # Calculate upper and lower percentiles for the PCE prior and posterior
    confidence_factor_95p = 1.96
    lower_percentile_PCE = bottom_μ_PCE - confidence_factor_95p * bottom_σ_PCE
    upper_percentile_PCE = bottom_μ_PCE + confidence_factor_95p * bottom_σ_PCE
    lower_percentile_post = bottom_μ_post - confidence_factor_95p * bottom_σ_post
    upper_percentile_post = bottom_μ_post + confidence_factor_95p * bottom_σ_post

    # Find all sensor points which lie on the bottom boundary
    bottom_ids = findall(x -> x[2] < 1.0e-12, sensor_points)

    bottom_ux = zeros(length(bottom_ids))
    bottom_x_sen = zeros(length(bottom_ids))
    for i in eachindex(bottom_ids)
        bottom_x_sen[i] = sensor_points[bottom_ids[i]][1]
        bottom_ux[i] = measurements[bottom_ids[i]][1]
    end


    # Finally, plot both prior and posterior with confidence bands
    band!(f[1, 1], bottom_x_grid, lower_percentile_PCE, upper_percentile_PCE; label = "prior 95% CI", color = (:red, 0.5))
    lines!(f[1, 1], bottom_x_grid, bottom_μ_PCE; label = "prior mean", color = :red)
    band!(f[1, 1], bottom_x_grid, lower_percentile_post, upper_percentile_post; label = "posterior 95% CI", color = (:blue, 0.5))
    lines!(f[1, 1], bottom_x_grid, bottom_μ_post; label = "posterior mean", color = :blue)
    # as well as the measurement data
    scatter!(f[1, 1], bottom_x_sen, bottom_ux; label = "noisy measurements", color = :black)
    axislegend(ax, position = :lt)

    return f
end

# Plot the underlying grid as well as the sensor points
function plot_sensors(grid, sensor_points::Vector{Vector{Float64}})

    sensorvis = GridVisualizer(; Plotter = CairoMakie)

    gridplot!(sensorvis, grid, show_colorbar = false)
    x = zeros(length(sensor_points))
    y = zeros(length(sensor_points))
    for i in eachindex(sensor_points)
        x[i] = sensor_points[i][1]
        y[i] = sensor_points[i][2]
    end
    customplot!(sensorvis) do ax
        CairoMakie.scatter!(ax, x, y)
    end
    return sensorvis
end

# This function generates the synthetic data based on a Mooney-Rivlin model.
# Note that the forward model has to be able to solve with this material model to
# regenerate the data.
# The data in the repository has been generated using the default,
# i.e. the Ferrite.jl-based implementation [here](https://github.com/statFEM-EUCLID/HolePlate2D_Ferrite.jl/)
function generate_synthetic_data(;
        F = 0.5,
        σ_sensor = 0.5,
        server_url = "http://localhost:4232"
    )

    # Note that we also need the projection model here
    # since we want to map the FEM solution to the given sensor points.
    model_name = "HolePlate2D"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)
    projection_model = UMBridge.HTTPModel(model_name * ".projection", server_url)

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

    # Then, we read in the sensor locations and obtain the projection matrix
    dataframe_sensors = CSV.read("Example201_sensors.csv", DataFrame)
    sensor_points = [Vector{Float64}(row) for row in eachrow(dataframe_sensors)]
    sensor_set = EqualSensorSet(sensor_points, σ_sensor)
    calculate_projection_matrix(sensor_set, projection_model, n_dofs)

    # and project the solution onto the sensor points
    n_sen = size(sensor_set)
    u_measured = zeros(n_sen, 2)
    u_measured[:, 1] = sensor_set.projection * u_truth[:, 1]
    u_measured[:, 2] = sensor_set.projection * u_truth[:, 2]

    # Finally, we sample a Gaussian noise at the sensor locations
    # and perturb the measurement data
    rng = MersenneTwister(2530)
    noise_distribution = Normal(0.0, σ_sensor)
    noise_x = rand(rng, noise_distribution, n_sen)
    noise_y = rand(rng, noise_distribution, n_sen)
    u_measured[:, 1] += noise_x
    u_measured[:, 2] += noise_y

    # before writing the result to file.
    dataframe_measurements = DataFrame(u_x = u_measured[:, 1], u_y = u_measured[:, 2])
    return CSV.write("Example201_measurements.csv", dataframe_measurements)
end


end
