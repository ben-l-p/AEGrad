from __future__ import annotations
from jax import Array
from typing import Sequence, Optional
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from aegrad.plotting.structured_grid import plot_frame_to_vtk
from aegrad.print_output import warn
from aegrad.algebra.array_utils import ArrayList


r"""
Everybody loves a dataclass
"""

@dataclass
class GridDiscretization:
    r"""
    Data class to hold grid discretization parameters
    - m: Number of panels in the chordwise direction
    - n: Number of panels in the spanwise direction
    - m_star: Number of wake panels in the chordwise direction
    """
    m: int
    n: int
    m_star: int

@dataclass
class LinearComponent:
    enabled: bool
    slices: Optional[Sequence[slice]]
    shapes: Optional[Sequence[tuple[int]]]

@dataclass
class _SliceEntry:
    name: str
    enabled: bool
    shapes: Optional[Sequence[tuple[int, ...]]]

@dataclass
class InputSlices:
    zeta_b: LinearComponent
    zeta_b_dot: LinearComponent
    nu_b: LinearComponent
    nu_w: LinearComponent

@dataclass
class StateSlices:
    gamma_b: LinearComponent
    gamma_w: LinearComponent
    gamma_bm1: LinearComponent
    zeta_w: LinearComponent
    zeta_b: LinearComponent

@dataclass
class OutputSlices:
    f_steady: LinearComponent
    f_unsteady: LinearComponent

@dataclass
class InputUnflattened:
    zeta_b: ArrayList
    zeta_b_dot: ArrayList
    nu_b: Optional[ArrayList]
    nu_w: Optional[ArrayList]

@dataclass
class StateUnflattened:
    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_bm1: Optional[ArrayList]
    zeta_w: Optional[ArrayList]
    zeta_b: Optional[ArrayList]

@dataclass
class OutputUnflattened:
    f_steady: ArrayList
    f_unsteady: Optional[ArrayList]


class AeroSnapshot:
    def __init__(self,
                 zeta_b: ArrayList,
                zeta_b_dot: ArrayList,
                 zeta_w: ArrayList,
                 gamma_b: ArrayList,
                 gamma_w: ArrayList,
                 f_steady: ArrayList,
                 f_unsteady: ArrayList,
                 surf_b_names: Sequence[str],
                 surf_w_names: Sequence[str],
                 i_ts: int,
                 t: Array,
                 n_surf: int) -> None:
        r"""
        Snapshot of aerodynamic surface at a single time step
        :param zeta_b: Bound grid coordinates, [n_surf][m+1, n+1, 3]
        :param zeta_b_dot: Bound grid velocities, [n_surf][m+1, n+1, 3]
        :param zeta_w: Wake grid coordinates, [n_surf][m_star+1, n+1, 3]
        :param gamma_b: Bound circulation strengths, [n_surf][m, n]
        :param gamma_w: Wake circulation strengths, [n_surf][m_star, n]
        :param f_steady: Steady force contributions, [n_surf][m, n]
        :param f_unsteady: Unsteady force contributions, [n_surf][m, n]
        """
        self.zeta_b: ArrayList = zeta_b
        self.zeta_b_dot: ArrayList = zeta_b_dot
        self.zeta_w: ArrayList = zeta_w
        self.gamma_b: ArrayList = gamma_b
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: ArrayList = f_unsteady
        self.surf_b_names:  Sequence[str] = surf_b_names
        self.surf_w_names:  Sequence[str] = surf_w_names
        self.i_ts: int = i_ts
        self.t: Array = t
        self.n_surf: int = n_surf

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        if i_surf < 0 or i_surf >= self.n_surf:
            raise IndexError("AeroSnapshot index out of range")

        return AeroSurfaceSnapshot(
            zeta_b=self.zeta_b[i_surf],
            zeta_b_dot=self.zeta_b_dot[i_surf],
            zeta_w=self.zeta_w[i_surf],
            gamma_b=self.gamma_b[i_surf],
            gamma_w=self.gamma_w[i_surf],
            f_steady=self.f_steady[i_surf],
            f_unsteady=self.f_unsteady[i_surf],
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=self.i_ts,
            t=self.t,
        )

    def plot(self, directory: PathLike, plot_wake: bool = True) -> Sequence[Path]:
        paths = []
        for i_surf in range(self.n_surf):
            paths.extend(self[i_surf].plot(directory, plot_bound=True, plot_wake=plot_wake))
        return paths

@dataclass
class AeroSurfaceSnapshot:
    def __init__(self,
                 zeta_b: Array,
                zeta_b_dot: Array,
                 zeta_w: Array,
                 gamma_b: Array,
                 gamma_w: Array,
                 f_steady: Array,
                 f_unsteady: Array,
                 surf_b_name: str,
                 surf_w_name: str,
                 i_ts: int,
                 t: Array) -> None:
        self.zeta_b: Array = zeta_b
        self.zeta_b_dot: Array = zeta_b_dot
        self.zeta_w: Array = zeta_w
        self.gamma_b: Array = gamma_b
        self.gamma_w: Array = gamma_w
        self.f_steady: Array = f_steady
        self.f_unsteady: Array = f_unsteady
        self.surf_b_name: str = surf_b_name
        self.surf_w_name: str = surf_w_name
        self.i_ts: int = i_ts
        self.t: Array = t

    def plot(self, directory: PathLike, plot_bound: bool = True, plot_wake: bool = True) -> Sequence[Path]:
        paths = []
        if plot_bound:
            bound_filename = Path(directory).joinpath(self.surf_b_name)
            paths.append(plot_frame_to_vtk(self.zeta_b, bound_filename, self.i_ts,
                              node_vector_data={'f_steady': self.f_steady, 'f_unsteady': self.f_unsteady,
                                                'zeta_dot': self.zeta_b_dot},
                              cell_scalar_data={'gamma': self.gamma_b},
                              ))
        if plot_wake:
            if self.gamma_w.shape[0] == 0:
                warn("No wake panels to plot, skipping.")
            else:
                wake_filename = Path(directory).joinpath(self.surf_w_name)
                paths.append(plot_frame_to_vtk(self.zeta_w, wake_filename, self.i_ts,
                                  cell_scalar_data={'gamma': self.gamma_w},
                                  ))
        return paths