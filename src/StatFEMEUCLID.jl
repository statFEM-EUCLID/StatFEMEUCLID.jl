# Copyright (c) 2025-2026 Jan Philipp Thiele
# SPDX-License-Identifier: MIT

module StatFEMEUCLID
using Reexport

include("FEMClient.jl")
@reexport using .FEMClient
include("Sampling.jl")
@reexport using .Sampling
include("PCE.jl")
@reexport using .PCE
include("SensorSets.jl")
@reexport using .SensorSets
include("StatFEM.jl")
@reexport using .StatFEM


end
