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

@testset "StatFEMEUCLID.jl" begin
    @testset "1DBar Example" begin
        # runic: off
##
        # runic: on
        N = 50
        function solution(F, E = 200, A = 20)
            return @. [LinRange(0, 100, N) * F / (A * E)]
        end
        p1 = @patch UMBridge.evaluate(model, input, config = Dict()) = solution(only(only(input)))
        p2 = @patch UMBridge.model_output_sizes(model::HTTPModel, config = Dict()) = (N,)

        rng = MersenneTwister(2020)  #fixed seed for comparability between runs
        μ_F = 800.0
        σ_F = 0.1
        n_MonteCarlo = 1000
        n_PCE = 50
        server_url = "http://localhost:4343"

        fem_model = UMBridge.HTTPModel("Bar1D.FEM", server_url)

        dist = Normal(μ_F, σ_F)
        sample_MC = apply((p1, p2)) do
            StatFEMEUCLID.Sampling.sample_FEM(fem_model, n_MonteCarlo, sample_distribution = dist, rng = rng)
        end
        pce_surrogate = apply((p1, p2)) do
            sample_PCE = StatFEMEUCLID.Sampling.sample_FEM(fem_model, n_PCE, sample_distribution = dist, rng = rng)
            StatFEMEUCLID.PCE.PolyChaosExpansion(sample_PCE)
        end
        mu, _ = StatFEMEUCLID.Sampling.compute_statistics(sample_MC)
        μ_PCE, σ_PCE = StatFEMEUCLID.PCE.compute_statistics(pce_surrogate)
        @test isapprox(mu[end], 20, rtol = 5e-7)
        @test isapprox(μ_PCE[end], 20, rtol = 5e-12)
        # runic: off
##
        # runic: on
    end
end
