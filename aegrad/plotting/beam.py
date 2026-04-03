from typing import Optional
import os
from pathlib import Path
from functools import partial

import vtk
from vtk.numpy_interface import algorithms as algs  # type: ignore
from vtk.numpy_interface import dataset_adapter as dsa  # type: ignore
from print_utils import warn

import jax
from jax import Array
from jax import numpy as jnp

from algebra.se3 import hg_to_d, exp_se3, hg_inv


def interpolate_beam(
        hg1: Array, hg2: Array, o0: Array, n_interp: int, include_endpoints: bool = False
) -> Array:
    """
    Interpolate beam geometry and orientation between two nodes.
    :param hg1: SE(3) transform of node 1, [4, 4]
    :param hg2: SE(3) transform of node 2, [4, 4]
    :param o0: Local beam orientation transformation, [3, 3]
    :param n_interp: Number of interpolation points to compute.
    :param include_endpoints: Whether to include the original nodes in the output (if False, only the interpolated points are returned)
    :return: Interpolated SE(3) transforms along the beam, [n_interp, 4, 4]
    """
    # beamwise coordinates, [n_interp]
    s_l = (
        jnp.linspace(0.0, 1.0, n_interp)
        if include_endpoints
        else jnp.linspace(0.0, 1.0, n_interp + 2)[1:-1]
    )

    h0 = jnp.zeros((4, 4))
    h0 = h0.at[:3, :3].set(o0)
    h0 = h0.at[3, 3].set(1.0)

    hg1h0 = hg1 @ h0
    hg2h0 = hg2 @ h0

    d = hg_to_d(hg1h0, hg2h0)  # [6] twist coordinates from node 1 to node 2

    s_l_d = jnp.outer(s_l, d)  # [n_interp, 6] scaled twist coordinates along the beam

    exp_s_l_d = jax.vmap(exp_se3, 0, 0)(s_l_d)  # [n_interp, 4, 4]
    return jnp.einsum(
        "ij,hjk,kl->hil", hg1h0, exp_s_l_d, hg_inv(h0)
    )  # [n_interp, 4, 4]


def create_beam_unstructured_grid(
        hg: Array,
        conn: Array,
        o0: Array,
        n_interp: int,
) -> tuple[vtk.vtkUnstructuredGrid, Optional[Array], Optional[Array]]:
    """
    Create a VTK UnstructuredGrid representing line (beam) elements.
    :param hg: Array of node SE(3) transformations, [n_nodes_, 4, 4]
    :param conn: Connectivity array with shape, [n_elems, 2]
    :param o0: Array of local beam orientation transformations, [n_elems, 3, 3]
    :param n_interp: Number of interpolation points to add along each beam element (does not include endpoints)
    :return: vtkUnstructuredGrid with VTK_LINE cells, array of SE(3) transforms interpolated case, and element mapping
    array (mapping each new interpolated element to the original element index)
    """
    n_nodes = hg.shape[0]
    n_elems = conn.shape[0]

    coords = hg[
        :, :3, 3
    ]  # extract node coordinates from SE(3) transforms, [n_nodes_, 3]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a 2D array with shape (n_nodes_, 3)")

    conn = jnp.asarray(conn)
    if conn.ndim != 2 or conn.shape[1] != 2:
        raise ValueError("conn must be a 2D array with shape (n_elems, 2)")

    ug = vtk.vtkUnstructuredGrid()

    if n_interp > 0:
        # interpolate elements, use polyline elements
        interp_hg = jax.vmap(
            partial(interpolate_beam, n_interp=n_interp, include_endpoints=False),
            (0, 0, 0),
            0,
        )(hg[conn[:, 0], ...], hg[conn[:, 1], ...], o0)  # [n_elems, n_interp, 4, 4]

        full_hg = jnp.concatenate(
            (
                hg[conn[:, 0], ...][:, None, :, :],
                interp_hg,
                hg[conn[:, 1], ...][:, None, :, :],
            ),
            axis=1,
        )  # [n_elems, 2 + n_interp, 4, 4]

        added_nodes = jnp.arange(n_nodes, n_nodes + n_interp * n_elems).reshape(
            n_elems, n_interp
        )  # [n_elems, n_interp]

        interp_conns = jnp.concatenate(
            (conn[:, [0]], added_nodes, conn[:, [1]]), axis=1
        )  # [n_elems, 2 + n_interp]

        interp_coords = jnp.concatenate(
            (coords, interp_hg[:, :, :3, 3].reshape(-1, 3)), axis=0
        )  # [n_nodes_ + n_interp * n_elems, 3])

        # add points
        points = vtk.vtkPoints()
        pts_vec = algs.make_vector(
            interp_coords[:, 0].ravel(),
            interp_coords[:, 1].ravel(),
            interp_coords[:, 2].ravel(),
        )
        points.SetData(dsa.numpyTovtkDataArray(pts_vec, "Points"))
        ug.SetPoints(points)

        # build connectivity: for VTK unstructured grid, cells need an offsets/connectivity array or use vtkCellArray
        cell_array = vtk.vtkCellArray()
        # iterate elements and insert lines
        for i in range(n_elems):
            node_ids = interp_conns[i, :]
            cell_array.InsertNextCell(
                2 + n_interp
            )  # [2 + n_interp] node ids for this element, including original nodes and interpolated nodes
            for j in range(len(node_ids)):
                cell_array.InsertCellPoint(int(node_ids[j]))

        # create element mapping
        elem_map = (
            jnp.arange(n_elems)[:, None].repeat(1 + n_interp, axis=1).ravel()
        )  # one entry for each new element

        ug.SetCells(vtk.VTK_POLY_LINE, cell_array)
        return ug, full_hg, elem_map
    else:
        # no interpolation, just use original nodes and connectivity with line elements
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
        return ug, None, None


def plot_beam_to_vtk(
        hg: Array,
        conn: Array,
        o0: Array,
        n_interp: int,
        filename: str | os.PathLike,
        i_ts: Optional[int] = None,
        node_scalar_data: Optional[dict[str, Optional[Array]]] = None,
        node_vector_data: Optional[dict[str, Optional[Array]]] = None,
        cell_scalar_data: Optional[dict[str, Optional[Array]]] = None,
        cell_vector_data: Optional[dict[str, Optional[Array]]] = None,
) -> Path:
    """
    Write beam (line element) data to a VTU file.

    :param hg: Array of SE(3) elements, [n_nodes_, 4, 4]
    :param conn: Connectivity array, [n_elem, 2]
    :param o0: Array of local beam orientation transformations, [n_elem, 3, 3]
    :param n_interp: Number of interpolation points to add along each beam element (does not include endpoints)
    :param filename: Base filename (directory + base name); _ts_<i_ts> will be appended if i_ts provided
    :param i_ts: Optional time step index to append to filename
    :param node_scalar_data: dict of [name, [n_nodes_]]
    :param node_vector_data: dict of [name, [n_nodes_, 3]]
    :param cell_scalar_data: dict of [name, [n_elems]]
    :param cell_vector_data: dict of [name, [n_elems, 3]]
    :return: Path of the written VTU file
    """

    if n_interp < 0:
        warn(
            "Number of interpolation points cannot be negative; defaulting to 0 (no interpolation)"
        )
        n_interp = 0

    filepath = Path(filename)

    n_nodes = hg.shape[0]
    n_elems = conn.shape[0]

    ug, hg_interp, elem_map = create_beam_unstructured_grid(hg, conn, o0, n_interp)

    # attach node (point) data
    if node_scalar_data is not None:
        for name, arr in node_scalar_data.items():
            if arr is None:
                continue
            if arr.shape[0] != n_nodes:
                raise ValueError(
                    f"Node scalar '{name}' has incorrect length {arr.shape[0]}; expected {n_nodes}"
                )
            if n_interp > 0:  # add zero entries for interpolated case
                arr = jnp.zeros(n_nodes + n_interp * n_elems).at[:n_nodes].set(arr)
            ug.GetPointData().AddArray(dsa.numpyTovtkDataArray(arr.ravel(), name))

    if node_vector_data is not None:
        for name, arr in node_vector_data.items():
            if arr is None:
                continue
            if arr.shape != (n_nodes, 3):
                raise ValueError(
                    f"Node vector '{name}' must have shape {(n_nodes, 3)}, got {arr.shape}"
                )
            if n_interp > 0:  # add zero entries for interpolated case
                arr = (
                    jnp.zeros((n_nodes + n_interp * n_elems, 3))
                    .at[:n_nodes, :]
                    .set(arr)
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
            if arr.shape != (n_elems,):
                raise ValueError(
                    f"Cell scalar '{name}' has incorrect shape {arr.shape}; expected {(n_elems,)}"
                )
            ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(arr.ravel(), name))

    if cell_vector_data is not None:
        for name, arr in cell_vector_data.items():
            if arr is None:
                continue
            if arr.shape != (n_elems, 3):
                raise ValueError(
                    f"Cell vector '{name}' must have shape {(n_elems, 3)}, got {arr.shape}"
                )
            vectors = algs.make_vector(
                arr[:, 0].ravel(), arr[:, 1].ravel(), arr[:, 2].ravel()
            )
            ug.GetCellData().AddArray(dsa.numpyTovtkDataArray(vectors, name))

    # write to file
    name = filepath.name
    if i_ts is not None:
        name += f"_ts_{i_ts}"
    filename_full = Path(filepath.parent).joinpath(name).with_suffix(".vtu")

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(filename_full))
    writer.SetInputData(ug)
    writer.Write()

    return Path(filename_full)
