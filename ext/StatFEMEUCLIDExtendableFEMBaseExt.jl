module StatFEMEUCLIDExtendableFEMBaseExt

using StatFEMEUCLID

import StatFEMEUCLID: SensorSets.calculate_projection_matrix


using ExtendableGrids: ExtendableGrid, CellFinder, gFindLocal!, Coordinates, num_nodes, dim_grid, coord_type, Edge1D, Triangle2D

using ExtendableFEMBase: get_basis, ON_CELLS, H1Pk, FESpace, CellDofs


function locate_cell(grid::ExtendableGrid, point::Vector{T}) where {T <: AbstractFloat}
    reference_point = zeros(coord_type(grid), size(grid[Coordinates], 1))
    cell_finder = CellFinder(grid)
    cell_id = gFindLocal!(reference_point, cell_finder, point)
    return cell_id, reference_point
end

function local_basis_values(xref::Vector{T}) where {T <: AbstractFloat}
    if length(xref) == 1
        return local_basis_values_1D(xref)
    elseif length(xref) == 2
        return local_basis_values_2D(xref)
    end
end

function local_basis_values_1D(xref::Vector{T}) where {T <: AbstractFloat}
    basisvalues = zeros(T, 2, 1)
    get_basis(ON_CELLS, H1Pk{1, 1, 1}, Edge1D)(basisvalues, xref)
    return basisvalues[:, 1]
end

function local_basis_values_2D(xref::Vector{T}) where {T <: AbstractFloat}
    basisvalues = zeros(T, 3, 1)
    get_basis(ON_CELLS, H1Pk{1, 2, 1}, Triangle2D)(basisvalues, xref)
    return basisvalues[:, 1]
end

function global_node_ids(grid::ExtendableGrid, cell_id)
    FES = FESpace{H1Pk{1, dim_grid(grid), 1}}(grid)
    return FES[CellDofs][:, cell_id]
end

function cell_mapping(grid::ExtendableGrid, point::Vector{T}) where {T <: AbstractFloat}
    cell_id, reference_point = locate_cell(grid, point)
    return global_node_ids(grid, cell_id), local_basis_values(reference_point)
end


function StatFEMEUCLID.SensorSets.calculate_projection_matrix(sensor_set::EqualSensorSet{T}, grid::ExtendableGrid) where {T <: AbstractFloat}
    n_dof = num_nodes(grid)
    n_points = length(sensor_set.locations)
    sensor_set.projection = zeros(n_points, n_dof)
    for i in eachindex(sensor_set.locations)
        global_node_ids, basisvalues = cell_mapping(grid, sensor_set.locations[i])
        sensor_set.projection[i, global_node_ids] = basisvalues
    end
    return
end


end
