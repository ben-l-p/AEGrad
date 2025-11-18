from __future__ import annotations
from jax import Array
from typing import Sequence
from dataclasses import dataclass
from jax import numpy as jnp
from os import PathLike
from pathlib import Path
from aegrad.plotting.structured_grid import plot_frame_to_vtk


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

def make_rectangular_grid(m: int, n: int, chord: float, ea: float) -> Array:
    r"""
    Create a rectangular grid of points in the yz-plane
    :param m: Number of panels in the chordwise direction
    :param n: Number of panels in the spanwise direction
    :param chord: Total chord of the surface
    :param ea: Elastic axis location as fraction of chord
    :return: Array of shape [m+1, n+1, 3] representing grid points in 3D space
    """

    grid = jnp.zeros((m+1, n+1, 3))
    return grid.at[..., 0].set((jnp.linspace(0.0, chord, m + 1) - ea * chord)[:, None])


class AeroSnapshot:
    def __init__(self,
                 zeta_b: Sequence[Array],
                 zeta_w: Sequence[Array],
                 gamma_b: Sequence[Array],
                 gamma_w: Sequence[Array],
                 f_steady: Sequence[Array],
                 f_unsteady: Sequence[Array],
                 surf_b_names: Sequence[str],
                 surf_w_names: Sequence[str],
                 i_ts: int,
                 t: float,
                 n_surf: int) -> None:
        r"""
        Snapshot of aerodynamic surface at a single time step
        :param zeta_b: Bound grid coordinates, [n_surf][m+1, n+1, 3]
        :param zeta_w: Wake grid coordinates, [n_surf][m_star+1, n+1, 3]
        :param gamma_b: Bound circulation strengths, [n_surf][m, n]
        :param gamma_w: Wake circulation strengths, [n_surf][m_star, n]
        :param f_steady: Steady force contributions, [n_surf][m, n]
        :param f_unsteady: Unsteady force contributions, [n_surf][m, n]
        """
        self.zeta_b: Sequence[Array] = zeta_b
        self.zeta_w: Sequence[Array] = zeta_w
        self.gamma_b: Sequence[Array] = gamma_b
        self.gamma_w: Sequence[Array] = gamma_w
        self.f_steady: Sequence[Array] = f_steady
        self.f_unsteady: Sequence[Array] = f_unsteady
        self.surf_b_names:  Sequence[str] = surf_b_names
        self.surf_w_names:  Sequence[str] = surf_w_names
        self.i_ts: int = i_ts
        self.t: float = t
        self.n_surf: int = n_surf

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        if i_surf < 0 or i_surf >= self.n_surf:
            raise IndexError("AeroSnapshot index out of range")

        return AeroSurfaceSnapshot(
            zeta_b=self.zeta_b[i_surf],
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
                 zeta_w: Array,
                 gamma_b: Array,
                 gamma_w: Array,
                 f_steady: Array,
                 f_unsteady: Array,
                 surf_b_name: str,
                 surf_w_name: str,
                 i_ts: int,
                 t: float) -> None:
        self.zeta_b: Array = zeta_b
        self.zeta_w: Array = zeta_w
        self.gamma_b: Array = gamma_b
        self.gamma_w: Array = gamma_w
        self.f_steady: Array = f_steady
        self.f_unsteady: Array = f_unsteady
        self.surf_b_name: str = surf_b_name
        self.surf_w_name: str = surf_w_name
        self.i_ts: int = i_ts
        self.t: float = t

    def plot(self, directory: PathLike, plot_bound: bool = True, plot_wake: bool = True) -> Sequence[Path]:
        paths = []
        if plot_bound:
            bound_filename = Path(directory).joinpath(self.surf_b_name)
            paths.append(plot_frame_to_vtk(self.zeta_b, bound_filename, self.i_ts,
                              node_vector_data={'f_steady': self.f_steady, 'f_unsteady': self.f_unsteady},
                              cell_scalar_data={'gamma': self.gamma_b},
                              ))
        if plot_wake:
            wake_filename = Path(directory).joinpath(self.surf_w_name)
            paths.append(plot_frame_to_vtk(self.zeta_w, wake_filename, self.i_ts,
                              cell_scalar_data={'gamma': self.gamma_w},
                              ))
        return paths
