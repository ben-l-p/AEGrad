from __future__ import annotations
from jax import Array
from typing import Sequence, Optional
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from aegrad.plotting.aerogrid import plot_frame_to_vtk
from aegrad.print_output import warn
from aegrad.algebra.array_utils import ArrayList


r"""
Everybody loves a dataclass
"""


@dataclass
class GridDiscretization:
    r"""
    Data class to hold grid discretization parameters
    :param m: Number of panels in the chordwise direction
    :param n: Number of panels in the spanwise direction
    :param m_star: Number of wake panels in the chordwise direction
    """

    m: int
    n: int
    m_star: int


@dataclass
class _LinearComponent:
    r"""
    Data class to hold information about components of a linear system.
    :param enabled: Whether this component is enabled
    :param slices: Optional sequence of slices to extract this component from a flattened array
    :param shapes: Optional sequence of shapes for unflattening this component, [n_surf][m, n, ...]
    """

    enabled: bool
    slices: Optional[Sequence[slice]]
    shapes: Optional[Sequence[tuple[int, ...]]]


@dataclass
class _SliceEntry:
    r"""
    Data class to hold information about a single slice entry
    :param name: Name of the component
    :param enabled: Whether this component is enabled
    :param shapes: Optional sequence of shapes for unflattening this component, [n_surf][m, n, ...]
    """

    name: str
    enabled: bool
    shapes: Optional[Sequence[tuple[int, ...]]]


@dataclass
class InputSlices:
    r"""
    Data class to hold linear components for input components
    :param zeta_b: Slices for bound grid coordinates
    :param zeta_b_dot: Slices for bound grid velocities
    :param nu_b: Slices for bound upwash
    :param nu_w: Slices for wake upwash
    """

    zeta_b: _LinearComponent
    zeta_b_dot: _LinearComponent
    nu_b: _LinearComponent
    nu_w: _LinearComponent


@dataclass
class StateSlices:
    r"""
    Data class to hold linear components for state components
    :param gamma_b: Slices for bound circulation strengths
    :param gamma_w: Slices for wake circulation strengths
    :param gamma_bm1: Slices for previous time step bound circulation strengths
    :param gamma_b_dot: Slices for bound circulation time derivatives
    :param zeta_w: Slices for wake grid coordinates
    :param zeta_b: Slices for bound grid coordinates
    """

    gamma_b: _LinearComponent
    gamma_w: _LinearComponent
    gamma_bm1: _LinearComponent
    gamma_b_dot: _LinearComponent
    zeta_w: _LinearComponent
    zeta_b: _LinearComponent


@dataclass
class OutputSlices:
    r"""
    Data class to hold linear components for output components
    :param f_steady: Slices for steady force contributions
    :param f_unsteady: Slices for unsteady force contributions
    """

    f_steady: _LinearComponent
    f_unsteady: _LinearComponent


@dataclass
class InputUnflattened:
    r"""
    Data class to hold unflattened input components, for either a single snapshot or a time series.
    :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param zeta_b_dot: Bound grid velocities, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param nu_b: Bound upwash, [n_surf][m, n, 3] or [n_surf][n_ts, m, n, 3]
    :param nu_w: Wake upwash, [n_surf][m_star, n, 3] or [n_surf][n_ts, m_star, n, 3]
    """

    zeta_b: ArrayList
    zeta_b_dot: ArrayList
    nu_b: Optional[ArrayList]
    nu_w: Optional[ArrayList]


@dataclass
class StateUnflattened:
    r"""
    Data class to hold unflattened state components, for either a single snapshot or a time series.
    :param gamma_b: Bound circulation strengths, [n_surf][m, n] or [n_surf][n_ts, m, n]
    :param gamma_w: Wake circulation strengths, [n_surf][m_star, n] or [n_surf][n_ts, m_star, n]
    :param gamma_bm1: Previous time step bound circulation strengths, [n_surf][m, n] or [n_surf][n_ts, m, n]
    :param gamma_b_dot: Bound circulation time derivatives, [n_surf][m, n] or [n_surf][n_ts, m, n]
    :param zeta_w: Wake grid coordinates, [n_surf][zeta_m_star, zeta_n, 3] or [n_surf][n_ts, zeta_m_star, zeta_n, 3]
    :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    """

    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_bm1: Optional[ArrayList]
    gamma_b_dot: Optional[ArrayList]
    zeta_w: Optional[ArrayList]
    zeta_b: Optional[ArrayList]


@dataclass
class OutputUnflattened:
    r"""
    Data class to hold unflattened output components, for either a single snapshot or a time series.
    :param f_steady: Steady force contributions, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    :param f_unsteady: Unsteady force contributions, [n_surf][zeta_m, zeta_n, 3] or [n_surf][n_ts, zeta_m, zeta_n, 3]
    """

    f_steady: ArrayList
    f_unsteady: Optional[ArrayList]


class AeroSnapshot:
    r"""
    Class to hold snapshot of multiple aerodynamic surfaces at a single time step.
    """

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: ArrayList,
        gamma_b: ArrayList,
        gamma_b_dot: ArrayList,
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: ArrayList,
        surf_b_names: Sequence[str],
        surf_w_names: Sequence[str],
        i_ts: int,
        t: Array,
        n_surf: int,
    ) -> None:
        r"""
        Snapshot of multiple aerodynamic surfaces at a single time step
        :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3]
        :param zeta_b_dot: Bound grid velocities, [n_surf][zeta_m, zeta_n, 3]
        :param zeta_w: Wake grid coordinates, [n_surf][zeta_m_star, zeta_n, 3]
        :param gamma_b: Bound circulation strengths, [n_surf][m, n]
        :param gamma_w: Wake circulation strengths, [n_surf][m_star, n]
        :param f_steady: Steady force contributions, [n_surf][zeta_m, zeta_n, 3]
        :param f_unsteady: Unsteady force contributions, [n_surf][zeta_m, zeta_n, 3]
        :param surf_b_names: Names of bound surfaces, [n_surf]
        :param surf_w_names: Names of wake surfaces, [n_surf]
        :param i_ts: Time step index
        :param t: Time at this snapshot
        :param n_surf: Number of aerodynamic surfaces
        """
        self.zeta_b: ArrayList = zeta_b
        self.zeta_b_dot: ArrayList = zeta_b_dot
        self.zeta_w: ArrayList = zeta_w
        self.gamma_b: ArrayList = gamma_b
        self.gamma_b_dot: ArrayList = gamma_b_dot
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: ArrayList = f_unsteady
        self.surf_b_names: Sequence[str] = surf_b_names
        self.surf_w_names: Sequence[str] = surf_w_names
        self.i_ts: int = i_ts
        self.t: Array = t
        self.n_surf: int = n_surf

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        r"""
        Get snapshot for a single aerodynamic surface.
        :param i_surf: Index of aerodynamic surface
        :return: AeroSurfaceSnapshot for the specified surface
        """
        if i_surf < 0 or i_surf >= self.n_surf:
            raise IndexError("AeroSnapshot index out of range")

        return AeroSurfaceSnapshot(
            zeta_b=self.zeta_b[i_surf],
            zeta_b_dot=self.zeta_b_dot[i_surf],
            zeta_w=self.zeta_w[i_surf],
            gamma_b=self.gamma_b[i_surf],
            gamma_b_dot=self.gamma_b_dot[i_surf],
            gamma_w=self.gamma_w[i_surf],
            f_steady=self.f_steady[i_surf],
            f_unsteady=self.f_unsteady[i_surf],
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=self.i_ts,
            t=self.t,
        )

    def plot(self, directory: PathLike, plot_wake: bool = True) -> Sequence[Path]:
        r"""
        Plot all aerodynamic surfaces in the snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """
        paths = []
        for i_surf in range(self.n_surf):
            paths.extend(
                self[i_surf].plot(directory, plot_bound=True, plot_wake=plot_wake)
            )
        return paths


@dataclass
class AeroSurfaceSnapshot:
    r"""
    Data class to hold snapshot of a single aerodynamic surface at a single time step.
    """

    def __init__(
        self,
        zeta_b: Array,
        zeta_b_dot: Array,
        zeta_w: Array,
        gamma_b: Array,
        gamma_b_dot: Array,
        gamma_w: Array,
        f_steady: Array,
        f_unsteady: Array,
        surf_b_name: str,
        surf_w_name: str,
        i_ts: int,
        t: Array,
    ) -> None:
        r"""
        Snapshot of an aerodynamic surface at a single time step
        :param zeta_b: Bound grid coordinates, [zeta_m, zeta_n, 3]
        :param zeta_b_dot: Bound grid velocities, [zeta_m, zeta_n, 3]
        :param zeta_w: Wake grid coordinates, [zeta_m_star, zeta_n, 3]
        :param gamma_b: Bound circulation strengths, [m, n]
        :param gamma_w: Wake circulation strengths, [m_star, n]
        :param f_steady: Steady force contributions, [zeta_m, zeta_n, 3]
        :param f_unsteady: Unsteady force contributions, [zeta_m, zeta_n, 3]
        :param surf_b_name: Names of bound surface
        :param surf_w_name: Names of wake surface
        :param i_ts: Time step index
        :param t: Time at this snapshot
        """
        self.zeta_b: Array = zeta_b
        self.zeta_b_dot: Array = zeta_b_dot
        self.zeta_w: Array = zeta_w
        self.gamma_b: Array = gamma_b
        self.gamma_b_dot: Array = gamma_b_dot
        self.gamma_w: Array = gamma_w
        self.f_steady: Array = f_steady
        self.f_unsteady: Array = f_unsteady
        self.surf_b_name: str = surf_b_name
        self.surf_w_name: str = surf_w_name
        self.i_ts: int = i_ts
        self.t: Array = t

    def plot(
        self, directory: PathLike, plot_bound: bool = True, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot aerodynamic surface in the snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_bound: If True, plot the bound surface.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """
        paths = []
        if plot_bound:
            bound_filename = Path(directory).joinpath(self.surf_b_name)
            paths.append(
                plot_frame_to_vtk(
                    self.zeta_b,
                    bound_filename,
                    self.i_ts,
                    node_vector_data={
                        "f_steady": self.f_steady,
                        "f_unsteady": self.f_unsteady,
                        "zeta_dot": self.zeta_b_dot,
                    },
                    cell_scalar_data={
                        "gamma": self.gamma_b,
                        "gamma_dot": self.gamma_b_dot,
                    },
                )
            )
        if plot_wake:
            if not self.gamma_w.shape[0]:
                warn("No wake panels to plot, skipping.")
            else:
                wake_filename = Path(directory).joinpath(self.surf_w_name)
                paths.append(
                    plot_frame_to_vtk(
                        self.zeta_w,
                        wake_filename,
                        self.i_ts,
                        cell_scalar_data={"gamma": self.gamma_w},
                    )
                )
        return paths
