from __future__ import annotations

from typing import Any

import jax
import numpy as np
import vtk

# noinspection PyUnresolvedReferences
from vtk.util import numpy_support as vns

_VTK_ID_DTYPE = np.dtype(vns.get_numpy_array_type(vtk.VTK_ID_TYPE))


def to_host(arr: Any) -> np.ndarray:
    """Convert an array to a contiguous Numpy array."""
    return np.ascontiguousarray(np.asarray(jax.device_get(arr)))


def configure_fast_writer(writer: vtk.vtkXMLWriter) -> None:
    """Switch a vtkXMLWriter to raw appended binary with no compression."""
    writer.SetDataModeToAppended()
    writer.SetEncodeAppendedData(False)
    writer.SetCompressorTypeToNone()


def make_vtk_points(coords: np.ndarray) -> vtk.vtkPoints:
    """Build a ``vtkPoints`` object from an ``(n, 3)`` Numpy array."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(vns.numpy_to_vtk(coords, deep=True))
    return points


def make_line_cell_array(conn: np.ndarray) -> vtk.vtkCellArray:
    """Vectorised construction of a VTK_LINE cell array from ``(n_elem, 2)``."""
    conn = np.ascontiguousarray(conn, dtype=_VTK_ID_DTYPE)
    n_elem = conn.shape[0]
    offsets = np.arange(0, 2 * (n_elem + 1), 2, dtype=_VTK_ID_DTYPE)
    connectivity = conn.ravel()

    ca = vtk.vtkCellArray()
    ca.SetData(
        vns.numpy_to_vtkIdTypeArray(offsets, deep=True),
        vns.numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    return ca


def make_poly_line_cell_array(conn: np.ndarray) -> vtk.vtkCellArray:
    """Vectorised construction of a VTK_POLY_LINE cell array from a connectivity array."""
    conn = np.ascontiguousarray(conn, dtype=_VTK_ID_DTYPE)
    n_elem, k = conn.shape
    offsets = np.arange(0, k * (n_elem + 1), k, dtype=_VTK_ID_DTYPE)
    connectivity = conn.ravel()

    ca = vtk.vtkCellArray()
    ca.SetData(
        vns.numpy_to_vtkIdTypeArray(offsets, deep=True),
        vns.numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    return ca


def numpy_to_vtk_scalar(arr: np.ndarray, name: str) -> vtk.vtkDataArray:
    """Convert a Numpy vector to a named VTK data array."""
    arr = np.ascontiguousarray(arr).ravel()
    data = vns.numpy_to_vtk(arr, deep=True)
    data.SetName(name)
    return data


def numpy_to_vtk_vector(arr: np.ndarray, name: str) -> vtk.vtkDataArray:
    """Convert a ``(n, 3)`` Numpy array to a named 3-component VTK data array."""
    arr = np.ascontiguousarray(arr).reshape(-1, 3)
    data = vns.numpy_to_vtk(arr, deep=True)
    data.SetName(name)
    return data
