using StatFEMEUCLID
using Test
using Aqua
using Mocking
using StatFEMEUCLID.PCE.Distributions
using StatFEMEUCLID.FEMClient.UMBridge

Aqua.test_all(StatFEMEUCLID)

Mocking.activate()

function create_lognormal_distribution(μ, σ)
    μ_log = log(μ^2 / sqrt(μ^2 + σ^2))
    σ_log = sqrt(log(1 + σ^2 / (μ^2)))
    return LogNormal(μ_log, σ_log)
end

@testset "StatFEMEUCLID.jl" begin
    @testset "1DBar Example" begin
        function solution(x, J, A, Y)
            return @. J / (A * Y) * x 
        end
        @patch UMBridge.evaluate(model, input, config=Dict()) = solution(input...)
        rng = MersenneTwister(2020)  #fixed seed for comparability between runs

        μ_E = 200.0
        σ_E = 10.0
        n_MonteCarlo = 100
    
        fem_model = UMBridge.HTTPModel("Bar1D.FEM", server_url)
        
        lognormal_dist = create_lognormal_distribution(μ_E, σ_E)
        sample_MC = StatFEMEUCLID.Sampling.sample_FEM(fem_model, n_MonteCarlo, sample_distribution = lognormal_dist, rng = rng)
    end
end
