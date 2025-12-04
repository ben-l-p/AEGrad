from typing import Optional
from os import PathLike

from jax import Array
from jax import numpy as jnp
from pathlib import Path

import vtk
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa


def create_structured_grid(grid_arr: Array, is_planar: bool) -> vtk.vtkStructuredGrid:
    # add a dummy third dimension of zeros if only 2D data is provided, where y=0
    sg = vtk.vtkStructuredGrid()
    if is_planar:
        sg.SetDimensions(*grid_arr.shape[:-1], 1)
    else:
        sg.SetDimensions(*grid_arr.shape[:-1])

    i_swap = 2 - int(is_planar)  # we swap axes as VTK likes z, y, x order

    # add point coordinate data
    points = vtk.vtkPoints()
    points_vec = algs.make_vector(
        *[jnp.swapaxes(grid_arr[..., i], 0, i_swap).ravel() for i in range(3)]
    )
    points.SetData(dsa.numpyTovtkDataArray(points_vec, "Points"))
    sg.SetPoints(points)

    return sg


def plot_frame_to_vtk(
    grid_arr: Array,
    filename: str | PathLike,
    i_ts: Optional[int] = None,
    node_scalar_data: Optional[dict[str, Array]] = None,
    node_vector_data: Optional[dict[str, Array]] = None,
    cell_scalar_data: Optional[dict[str, Array]] = None,
    cell_vector_data: Optional[dict[str, Array]] = None,
) -> Path:
    r"""
    Plot a single timestep of grid data
    :param grid_arr: Structured grid array with shapes (n_x, n_y, n_z, 2|3)
    :param filename: Base filename, including directory. Information on the frame number will be
    appended to this.
    :param i_ts: Timestep to write
    :param node_scalar_data: Dictionary of node scalar data
    :param node_vector_data: Dictionary of node vector data
    :param cell_scalar_data: Dictionary of cell scalar data
    :param cell_vector_data: Dictionary of cell vector data
    """

    # planar grid should have 3 dimensions, while volume grid should have 4 dimensions
    match grid_arr.ndim:
        case 3:
            is_planar = True
        case 4:
            is_planar = False
        case _:
            raise ValueError(
                f"grid_arr must have 3 or 4 dimensions, got {grid_arr.ndim}-D array"
            )

    i_swap = 2 - int(is_planar)  # we swap axes as VTK likes z, y, x order

    sg = create_structured_grid(grid_arr, is_planar)

    # cell scalar data
    if cell_scalar_data is not None:
        for name, arr in cell_scalar_data.items():
            sg.GetCellData().AddArray(
                dsa.numpyTovtkDataArray(jnp.swapaxes(arr, 0, i_swap).ravel(), name)
            )

    # cell vector data
    if cell_vector_data is not None:
        for name, arr in cell_vector_data.items():
            if arr.shape[-1] != 3:
                raise ValueError(
                    f"Node vector data '{name}' must have trailing dimension of size 3, got {arr.shape[-1]}"
                )

            vectors = algs.make_vector(
                jnp.swapaxes(arr[..., 0], 0, i_swap).ravel(),
                jnp.swapaxes(arr[..., 1], 0, i_swap).ravel(),
                jnp.swapaxes(arr[..., 2], 0, i_swap).ravel(),
            )
            sg.GetCellData().AddArray(dsa.numpyTovtkDataArray(vectors, name))

    # point scalar data
    if node_scalar_data is not None:
        for name, arr in node_scalar_data.items():
            sg.GetPointData().AddArray(
                dsa.numpyTovtkDataArray(jnp.swapaxes(arr, 0, i_swap).ravel(), name)
            )

    # point vector data
    if node_vector_data is not None:
        for name, arr in node_vector_data.items():
            if arr.shape[-1] != 3:
                raise ValueError(
                    f"Node vector data '{name}' must have trailing dimension of size 3, got {arr.shape[-1]}"
                )

            vectors = algs.make_vector(
                jnp.swapaxes(arr[..., 0], 0, i_swap).ravel(),
                jnp.swapaxes(arr[..., 1], 0, i_swap).ravel(),
                jnp.swapaxes(arr[..., 2], 0, i_swap).ravel(),
            )
            sg.GetPointData().AddArray(dsa.numpyTovtkDataArray(vectors, name))

    # write to file
    filename_full = (
        f"{filename}_ts_{i_ts}.vts" if i_ts is not None else f"{filename}.vts"
    )
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename_full)
    writer.SetInputData(sg)
    writer.Write()

    return Path(filename_full)
