using StatFEMEUCLID
using Test
using Aqua
using Mocking
using Distributions
using UMBridge
using UMBridge: HTTPModel
using Random

Aqua.test_all(StatFEMEUCLID)

Mocking.activate()

function create_lognormal_distribution(μ, σ)
    μ_log = log(μ^2 / sqrt(μ^2 + σ^2))
    σ_log = sqrt(log(1 + σ^2 / (μ^2)))
    return LogNormal(μ_log, σ_log)
end

@testset "StatFEMEUCLID.jl" begin
    @testset "1DBar Example" begin
        # runic: off
##
        # runic: on
        N = 50
        function solution(E, F = 800, A = 20)
            return [LinRange(0, 100, N) * F / (A * E)]
        end
        p1 = @patch UMBridge.evaluate(model, input, config = Dict()) = solution(only(input))
        p2 = @patch UMBridge.model_output_sizes(model::HTTPModel, config = Dict()) = (N,)

        rng = MersenneTwister(2020)  #fixed seed for comparability between runs
        μ_E = 200.0
        σ_E = 10.0
        n_MonteCarlo = 1000
        server_url = "http://localhost:4343"

        fem_model = UMBridge.HTTPModel("Bar1D.FEM", server_url)

        lognormal_dist = create_lognormal_distribution(μ_E, σ_E)
        sample_MC = apply(p1) do
            apply(p2) do
                StatFEMEUCLID.Sampling.sample_FEM(fem_model, n_MonteCarlo, sample_distribution = lognormal_dist, rng = rng)
            end
        end
        mu, _ = StatFEMEUCLID.Sampling.compute_statistics(sample_MC)
        @test isapprox(mu[end], 20, atol = 0.05)
        # runic: off
##
        # runic: on
    end
end
