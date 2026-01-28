from typing import Optional
from os import PathLike
from pathlib import Path

from jax import Array
from jax import numpy as jnp

from vtk.numpy_interface import algorithms as algs  # type: ignore
from vtk.numpy_interface import dataset_adapter as dsa  # type: ignore

import vtk


def create_beam_unstructured_grid(
    coords: Array,  # [..., 3] shaped array of node coordinates
    conn: Array,  # [n_elems, 2] array of node indices per line element
) -> vtk.vtkUnstructuredGrid:
    """
    Create a VTK UnstructuredGrid representing line (beam) elements.
    :param coords: Array of node coordinates with shape (n_nodes, 3) or (..., 3)
    :param conn: Connectivity array with shape (n_elems, 2) containing integer node indices
    :return: vtkUnstructuredGrid with VTK_LINE cells
    """
    coords = jnp.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a 2D array with shape (n_nodes, 3)")

    conn = jnp.asarray(conn)
    if conn.ndim != 2 or conn.shape[1] != 2:
        raise ValueError("conn must be a 2D array with shape (n_elems, 2)")

    n_elems = conn.shape[0]

    ug = vtk.vtkUnstructuredGrid()

    # add points
    points = vtk.vtkPoints()
    pts_vec = algs.make_vector(
        coords[:, 0].ravel(), coords[:, 1].ravel(), coords[:, 2].ravel()
    )
    points.SetData(dsa.numpyTovtkDataArray(pts_vec, "Points"))
    ug.SetPoints(points)

    # build connectivity: for VTK unstructured grid, cells need an offsets/connectivity array or use vtkCellArray
    cell_array = vtk.vtkCellArray()
    # iterate elements and insert lines
    for i in range(n_elems):
        n0 = int(conn[i, 0])
        n1 = int(conn[i, 1])
        cell_array.InsertNextCell(2)
        cell_array.InsertCellPoint(n0)
        cell_array.InsertCellPoint(n1)

    ug.SetCells(vtk.VTK_LINE, cell_array)

    return ug


def plot_beam_to_vtk(
    nodes: Array,
    conn: Array,
    filename: str | PathLike,
    i_ts: Optional[int] = None,
    node_scalar_data: Optional[dict[str, Array]] = None,
    node_vector_data: Optional[dict[str, Array]] = None,
    cell_scalar_data: Optional[dict[str, Array]] = None,
    cell_vector_data: Optional[dict[str, Array]] = None,
) -> Path:
    """
    Write beam (line element) data to a VTU file.

    :param nodes: Array of node coordinates, shape (n_nodes, 3)
    :param conn: Connectivity array, shape (n_elems, 2)
    :param filename: Base filename (directory + base name); _ts_<i_ts> will be appended if i_ts provided
    :param node_scalar_data: dict of [name, [n_nodes]]
    :param node_vector_data: dict of [name, [n_nodes, 3]]
    :param cell_scalar_data: dict of [name, [n_elems]]
    :param cell_vector_data: dict of [name, [n_elems, 3]]
    :return: Path of the written VTU file
    """
    filename = Path(filename)

    ug = create_beam_unstructured_grid(nodes, conn)

    # attach node (point) data
    if node_scalar_data is not None:
        for name, arr in node_scalar_data.items():
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            if arr.shape[0] != nodes.shape[0]:
                raise ValueError(
                    f"Node scalar '{name}' has incorrect length {arr.shape[0]}; expected {nodes.shape[0]}"
                )
            ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(arr.ravel(), name))

    if node_vector_data is not None:
        for name, arr in node_vector_data.items():
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            if arr.shape != nodes.shape:
                raise ValueError(
                    f"Node vector '{name}' must have shape {nodes.shape}, got {arr.shape}"
                )
            vectors = algs.make_vector(
                arr[:, 0].ravel(), arr[:, 1].ravel(), arr[:, 2].ravel()
            )
            ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(vectors, name))

    # attach cell data
    if cell_scalar_data is not None:
        for name, arr in cell_scalar_data.items():
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            if arr.shape[0] != conn.shape[0]:
                raise ValueError(
                    f"Cell scalar '{name}' has incorrect length {arr.shape[0]}; expected {conn.shape[0]}"
                )
            ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(arr.ravel(), name))

    if cell_vector_data is not None:
        for name, arr in cell_vector_data.items():
            if arr is None:
                continue
            arr = jnp.asarray(arr)
            if arr.shape[0] != conn.shape[0] or arr.shape[1] != 3:
                raise ValueError(
                    f"Cell vector '{name}' must have shape (n_elems,3), got {arr.shape}"
                )
            vectors = algs.make_vector(
                arr[:, 0].ravel(), arr[:, 1].ravel(), arr[:, 2].ravel()
            )
            ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(vectors, name))

    # write to file
    name = filename.name
    if i_ts is not None:
        name += f"_ts_{i_ts}"
    filename_full = Path(filename.parent).joinpath(name).with_suffix(".vtu")

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(filename_full))
    writer.SetInputData(ug)
    writer.Write()

    return Path(filename_full)
