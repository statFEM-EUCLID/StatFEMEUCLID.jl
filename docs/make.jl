push!(LOAD_PATH, "../src/")

using Documenter
using ExampleJuggler
using StatFEMEUCLID
using CairoMakie


function make_all(; with_examples::Bool = false, modules = :all, run_examples::Bool = false)
    module_examples = []
    if with_examples
        DocMeta.setdocmeta!(ExampleJuggler, :DocTestSetup, :(using ExampleJuggler); recursive = true)

        example_dir = joinpath(@__DIR__, "..", "examples")

        if modules === :all
            modules = filter(x -> endswith(x, ".jl"), readdir(example_dir))
        end

        cleanexamples()

        module_examples = @docmodules(example_dir, modules, Plotter = CairoMakie)
    end

    makedocs(
        modules = [StatFEMEUCLID, StatFEMEUCLID.FEMClient, StatFEMEUCLID.Sampling, StatFEMEUCLID.PCE],
        sitename = "StatFEMEUCLID.jl",
        authors = "Jan Philipp Thiele, Andrea Morini",
        format = Documenter.HTML(; repolink = "https://github.com/statFEM-EUCLID/StatFEMEUCLID.jl", mathengine = MathJax3()),
        clean = false,
        checkdocs = :none,
        warnonly = false,
        doctest = true,
        pages = [
            "Home" => "index.md"
            "UMBridge interfaces" => "umbridge_interfaces.md"
            "Submodules" => [
                "FEMClient" => "submodules/femclient.md"
                "Sampling" => "submodules/sampling.md"
                "PCE" => "submodules/pce.md"
                "SensorSets" => "submodules/sensorsets.md"
                "StatFEM" => "submodules/statfem.md"
                "Inverse" => "submodules/inverse.md"
            ]
            "Examples" => module_examples
        ],
    )

    return cleanexamples()
end

make_all(; with_examples = true)

deploydocs(
    repo = "github.com/statFEM-EUCLID/StatFEMEUCLID.jl/"
)
