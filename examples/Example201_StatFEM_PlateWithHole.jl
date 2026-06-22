# Copyright (c) 2026 Jan Philipp Thiele
# SPDX-License-Identifier MIT

#=

# 201: Using StatFEM on a 2D Plate with a hole
([source code](@__SOURCE_URL__))


In the example the statistical finite element method (StatFEM) 
is used to calculate a Bayesian posterior of a sampled finite element solution (prior) based on given data.
This articificial measurement data is obtained from a known set of hyperelastic material parameters $\kappa_\text{truth}$.
The prior is sampled using a perturbed/reduced set of material parameters.


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
using CSV,DataFrames


# With `server_url` you can point to a different UMBridge server implementing the model
# the default is what is provided by the Ferrite.jl-based implementation [here](https://github.com/statFEM-EUCLID/)

function main(;
        μ_F = 0.5,
        σ_F = 2.5e-2,
        n_PCE = 30,
        E= 1.35,
        ν= 0.35,
        σ_sensor = 1e-3,
        server_url = "http://localhost:4232",
    )

    rng = MersenneTwister(2030)  #fixed seed for comparability between runs

    # As in Example101 we start by defining UMBridge HTML models for querying the forward model
    # Additionally, we need the projection from degrees of freedom to sensor locations
    # with the projection model.
    model_name = "HolePlate2Dhyperelastic"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)
    projection_model = UMBridge.HTTPModel(model_name * ".projection",server_url)
    n_dofs = UMBridge.model_output_sizes(fem_model)[1]

    # Since some of our example servers can solve for more than one material model
    # we now need to specify this in a configuration
    LE_material_config = Dict{String,Any}(
        "material" => "LinElastic"
    )

    # Since we want to be able to change material parameters, (esp. in later examples)
    # we now use the keyword argument params to pass them
    ux_prior_sample = StatFEMEUCLID.Sampling.sample_FEM(
        fem_model,
        n_PCE,
        sample_distribution = Normal(μ_F, σ_F),
        rng = rng,
        config = LE_material_config,
        extra_params = [E,ν]
    )

    # For statFEM we need three components, a PCE surrogate, measurment data and 
    # a projection matrix H from our solution space (dofs) to the locations of the 
    # sensors where the measurements were taken.

    # 1) PCE surrogate, see Example101
    ux_pce_surrogate = StatFEMEUCLID.PCE.PolyChaosExpansion(ux_prior_sample)

    # 2) Sensor points from mesh + projection matrix


    dataframe_sensors = CSV.read("Example201_sensors.csv",DataFrame)
    sensor_points = [Vector{Float64}(row) for row in eachrow(dataframe_sensors)]

    sensor_set = StatFEMEUCLID.StatFEM.EqualSensorSet(sensor_points,σ_sensor)
    StatFEMEUCLID.StatFEM.get_projection_matrix(sensor_set, projection_model,n_dofs)

    # 3)  Read in (synthetic) measurements
    dataframe_measurements = CSV.read("Example201_measurements.csv",DataFrame)
    measurements = [Vector{Float64}(row) for row in eachrow(dataframe_measurements)]

    ux_measurements = zeros(length(measurements))
    for i in eachindex(measurements)
        ux_measurements[i] = measurements[i][1]
    end

    ux_posterior = StatFEMEUCLID.StatFEM.posterior_mean(sensor_set,ux_measurements,ux_pce_surrogate)

    grid = ExtendableGrids.simplexgrid_from_gmsh("Example201_mesh.msh")
    # Plot of sensor points and mesh
    # sensorvis = plot_sensors(grid, sensor_points)

    displt = plot_displacements_along_bottom_boundary(grid,sensor_points,measurements,ux_pce_surrogate,ux_posterior)


    # return sensorvis,displt
end

function plot_displacements_along_bottom_boundary(grid,sensor_points,measurements,pce_surrogate,posterior)
    f = Figure(size=(800,800))
    ax = Axis(f[1,1],xlabel = "x",ylabel="u_x")

    sub = subgrid(grid,[3],boundary=true)
    bottom_x_grid = Float64.(sub[Coordinates])[1,:]
    μ_PCE,σ_PCE = StatFEMEUCLID.PCE.compute_statistics(pce_surrogate)

    bottom_μ_PCE = μ_PCE[sub[NodeParents]]
    bottom_σ_PCE = σ_PCE[sub[NodeParents]] 
    bottom_posterior = posterior[sub[NodeParents]]


    confidence_factor_95p = 1.96
    lower_percentile = bottom_μ_PCE - confidence_factor_95p * bottom_σ_PCE
    upper_percentile = bottom_μ_PCE + confidence_factor_95p * bottom_σ_PCE
    
    bottom_ids = findall(x->x[2]<1.0e-12,sensor_points)

    bottom_ux = zeros(length(bottom_ids))
    bottom_x_sen = zeros(length(bottom_ids))
    for i in eachindex(bottom_ids)
      bottom_x_sen[i] = sensor_points[bottom_ids[i]][1]
      bottom_ux[i] = measurements[bottom_ids[i]][1]
    end


    band!(f[1,1],bottom_x_grid,lower_percentile,upper_percentile; label="prior 95% CI",color=(:red,0.5))
    lines!(f[1,1],bottom_x_grid,bottom_μ_PCE;label="prior mean",color=:red)
    lines!(f[1,1],bottom_x_grid,bottom_posterior;label="posterior mean",color=:blue)
    scatter!(f[1,1],bottom_x_sen,bottom_ux;label="noisy measurements",color=:black)
    axislegend(ax,position = :lt)

    return f
end


function plot_sensors(grid, sensor_points::Vector{Vector{Float64}})

    sensorvis = GridVisualizer(; Plotter = CairoMakie)

    gridplot!(sensorvis, grid,show_colorbar=false)
    x = zeros(length(sensor_points))
    y = zeros(length(sensor_points))
    for i in eachindex(sensor_points)
        x[i] = sensor_points[i][1]
        y[i] = sensor_points[i][2]
    end
    customplot!(ctx) do ax
        CairoMakie.scatter!(ax, sensor_points[])
    end
    return sensorvis
end


function generate_synthetic_data(;F = 0.5,server_url = "http://localhost:4040")
    model_name = "HolePlate2Dhyperelastic"
    fem_model = UMBridge.HTTPModel(model_name * ".FEM", server_url)
    projection_model = UMBridge.HTTPModel(model_name * ".projection",server_url)

    A_10 = 0.3
    A_01 = 0.2
    B_1 = 1.5

    #Workaround until inputSizes can be config dependent (PR is already opened)
    κ = zeros(Float64,3)
    κ[1] = A_10
    κ[2] = A_01
    κ[3] = B_1
    
    MR_material_config = Dict{String,Any}(
        "material" => "MooneyRivlin",
        "n_MR" => 1,
        "n_VOL" => 1
    )
    n_dofs = UMBridge.model_output_sizes(fem_model)[1]

    u_truth = zeros(n_dofs, 2)
    u_truth_full = StatFEMEUCLID.FEMClient.evaluate_fem_model(
        fem_model,F,solution_index=:,config=MR_material_config,extra_params=κ)
    u_truth[:,1] = u_truth_full[1:n_dofs]
    u_truth[:,2] = u_truth_full[n_dofs+1:end]
    σ_sensor = 1e-3

    dataframe_sensors = CSV.read("Example201_sensors.csv",DataFrame)
    sensor_points = [Vector{Float64}(row) for row in eachrow(dataframe_sensors)]

    sensor_set = StatFEMEUCLID.StatFEM.EqualSensorSet(sensor_points,σ_sensor)
    StatFEMEUCLID.StatFEM.get_projection_matrix(sensor_set,projection_model,n_dofs)

    n_sen = StatFEMEUCLID.StatFEM.size(sensor_set)
    u_measured = zeros(n_sen,2)
    u_measured[:,1] = sensor_set.projection*u_truth[:,1]
    u_measured[:,2] = sensor_set.projection*u_truth[:,2]

    rng = MersenneTwister(1530)

    #adding noise as function of sensor set?
    noise_distribution = Normal(0.,σ_sensor)
    noise_x = rand(rng,noise_distribution,n_sen)
    noise_y = rand(rng,noise_distribution,n_sen)
    u_measured[:,1]+= noise_x
    u_measured[:,2]+= noise_y

    dataframe_measurements = DataFrame(u_x = u_measured[:,1],u_y = u_measured[:,2])
    CSV.write("Example201_measurements.csv",dataframe_measurements)
end



end
