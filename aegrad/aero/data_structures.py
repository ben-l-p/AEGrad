from __future__ import annotations
from typing import Sequence, Optional
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import jax
from jax import numpy as jnp
from jax import Array

from aegrad.utils import _make_pytree
from aegrad.plotting.aerogrid import plot_frame_to_vtk
from aegrad.plotting.pvd import write_pvd
from aegrad.print_output import warn
from aegrad.algebra.array_utils import ArrayList, check_arr_shape
from aegrad.algebra.base import finite_difference
from aegrad.algebra.array_utils import split_to_vertex
from aero.flowfields import FlowField


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


@_make_pytree
class DynamicAero:
    r"""
    Class to hold time series of multiple aerodynamic surfaces.
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
        t: Array,
    ) -> None:
        r"""
        Time series of multiple aerodynamic surfaces
        :param zeta_b: Bound grid coordinates, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param zeta_b_dot: Bound grid velocities, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param zeta_w: Wake grid coordinates, [n_surf][n_ts, zeta_m_star, zeta_n, 3]
        :param gamma_b: Bound circulation strengths, [n_surf][n_ts, m, n]
        :param gamma_w: Wake circulation strengths, [n_surf][n_ts, m_star, n]
        :param f_steady: Steady force contributions, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param f_unsteady: Unsteady force contributions, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param surf_b_names: Names of bound surfaces, [n_surf]
        :param surf_w_names: Names of wake surfaces, [n_surf]
        :param t: Time array for the time series
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
        self.t: Array = t
        self.n_surf: int = len(surf_b_names)
        self.n_tstep: int = len(t)

    def __getitem__(self, i_ts: int) -> StaticAero:
        r"""
        Get snapshot for all aerodynamic surfaces.
        :param i_ts: Timestep index
        :return: StaticAero for the specified timestep
        """

        return StaticAero(
            zeta_b=self.zeta_b.index_all(i_ts),
            zeta_b_dot=self.zeta_b_dot.index_all(i_ts),
            zeta_w=self.zeta_w.index_all(i_ts),
            gamma_b=self.gamma_b.index_all(i_ts),
            gamma_b_dot=self.gamma_b_dot.index_all(i_ts),
            gamma_w=self.gamma_w.index_all(i_ts),
            f_steady=self.f_steady.index_all(i_ts),
            f_unsteady=self.f_unsteady.index_all(i_ts),
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t[i_ts],
            n_surf=self.n_surf,
        )

    def __setitem__(
        self,
        i_ts: int,
        snapshot: StaticAero,
    ) -> None:
        self.t = self.t.at[i_ts].set(snapshot.t)
        for i_surf in range(self.n_surf):
            self.zeta_b[i_surf] = (
                self.zeta_b[i_surf].at[i_ts, ...].set(snapshot.zeta_b[i_surf])
            )
            self.zeta_b_dot[i_surf] = (
                self.zeta_b_dot[i_surf].at[i_ts, ...].set(snapshot.zeta_b_dot[i_surf])
            )
            self.zeta_w[i_surf] = (
                self.zeta_w[i_surf].at[i_ts, ...].set(snapshot.zeta_w[i_surf])
            )
            self.gamma_b[i_surf] = (
                self.gamma_b[i_surf].at[i_ts, ...].set(snapshot.gamma_b[i_surf])
            )
            self.gamma_b_dot[i_surf] = (
                self.gamma_b_dot[i_surf].at[i_ts, ...].set(snapshot.gamma_b_dot[i_surf])
            )
            self.gamma_w[i_surf] = (
                self.gamma_w[i_surf].at[i_ts, ...].set(snapshot.gamma_w[i_surf])
            )
            self.f_steady[i_surf] = (
                self.f_steady[i_surf].at[i_ts, ...].set(snapshot.f_steady[i_surf])
            )
            self.f_unsteady[i_surf] = (
                self.f_unsteady[i_surf].at[i_ts, ...].set(snapshot.f_unsteady[i_surf])
            )

    def plot(
        self, directory: PathLike, plot_bound: bool = True, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot all aerodynamic surfaces in the time series snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """
        # TODO: add PVD file
        paths = []
        for i_ts in range(self.n_tstep):
            paths.extend(
                self[i_ts].plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)
            )
        return paths

    def get_zeta_te_surf(self, i_ts: int, i_surf: int) -> Array:
        r"""
        Get trailing edge grid coordinates for a single surface at specified time step
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :return: Trailing edge grid coordinates, [zeta_n, 3]
        """
        return self.zeta_b[i_surf][i_ts, -1, ...]

    def _get_gamma_te_surf(self, i_ts: int, i_surf: int) -> Array:
        r"""
        Get trailing edge circulation strengths for a single surface at specified time step
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :return: Trailing edge circulation strengths, [gamma_n]
        """
        return self.gamma_b[i_surf][i_ts, -1, :]

    def get_zeta_b(self, i_ts: int) -> ArrayList:
        r"""
        Get bound grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        return self.zeta_b.index_all(i_ts, ...)

    def get_zeta_dot_b(self, i_ts: int) -> ArrayList:
        r"""
        Get bound grid velocities for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        return self.zeta_b_dot.index_all(i_ts, ...)

    def get_zeta_w(self, i_ts: int) -> ArrayList:
        r"""
        Get wake grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][zeta_m_star, zeta_n, 3]
        """
        return self.zeta_w.index_all(i_ts, ...)

    def get_zeta(self, i_ts: int) -> ArrayList:
        r"""
        Get full grid coordinates all bound and wake surfaces at specified time step. These are stacked in (bound, wake)
        int_order.
        :param i_ts: Timestep index
        :return: List of grid coordinates for each surface, [2 * n_surf][zeta_m | zeta_m_star, zeta_n, 3]
        """
        return self.get_zeta_b(i_ts).combine(self.get_zeta_w(i_ts))

    def get_gamma_w(self, i_ts: int) -> ArrayList:
        r"""
        Get wake circulation strengths for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Wake circulation strengths for each surface, [n_surf][m_star, n]
        """
        return self.gamma_w.index_all(i_ts, ...)

    def get_gamma_vect(self, i_ts: int) -> Array:
        r"""
        Get total circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Total circulation strengths vector, [gamma_tot]
        """
        return jnp.concatenate(
            [self._get_gamma_b_vect(i_ts), self.get_gamma_w_vect(i_ts)], axis=0
        )

    def get_gamma_w_vect(self, i_ts: int) -> Array:
        r"""
        Get wake circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Wake circulation strengths vector, [gamma_w_tot]
        """
        return self.get_gamma_w(i_ts).flatten()

    def get_gamma_b(self, i_ts: int) -> ArrayList:
        r"""
        Get bound circulation strengths for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Bound circulation strengths for each surface, [n_surf][m, n]
        """
        return self.gamma_b.index_all(i_ts, ...)

    def _get_gamma_b_vect(self, i_ts: int) -> Array:
        r"""
        Get bound circulation strengths for all surfaces at specified time step as a vector
        :param i_ts: Timestep index
        :return: Bound circulation strength vector for all surfaces, [n_surf * m * n]
        """
        return self.get_gamma_b(i_ts).flatten()

    def set_zeta_b(self, zeta_b: ArrayList, i_ts: int) -> None:
        r"""
        Set bound grid coordinates from list of grid coordinates at specified time step
        :param zeta_b: List of bound grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b[i_surf] = self.zeta_b[i_surf].at[i_ts, ...].set(zeta_b[i_surf])

    def set_zeta_b_dot(self, zeta_b_dot: ArrayList, i_ts: int) -> None:
        r"""
        Set bound grid velocities from list of grid coordinates at specified time step
        :param zeta_b_dot: List of bound grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b_dot[i_surf] = (
                self.zeta_b_dot[i_surf].at[i_ts, ...].set(zeta_b_dot[i_surf])
            )

    def set_zeta_w(self, zeta_w: ArrayList, i_ts: int) -> None:
        r"""
        Set wake grid coordinates from list of grid coordinates at specified time step
        :param zeta_w: List of bound grid coordinates for each surface, [n_surf][zeta_m_star, zeta_n, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_w[i_surf] = self.zeta_w[i_surf].at[i_ts, ...].set(zeta_w[i_surf])

    def set_gamma_w_static(self, i_ts: int) -> None:
        r"""
        Set wake circulation strengths in static solution to all match trailing edge strengths.
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.gamma_w[i_surf] = (
                self.gamma_w[i_surf]
                .at[i_ts, ...]
                .set(self._get_gamma_te_surf(i_ts, i_surf)[None, :])
            )

    def calculate_gamma_dot(self, i_ts: int, dt: Array) -> None:
        r"""
        Calculate time derivative of bound circulation strengths at specified time step using finite difference.
        :param i_ts: Timestep index
        :param dt: Time step size
        """
        for i_surf in range(self.n_surf):
            self.gamma_b_dot[i_surf] = (
                self.gamma_b_dot[i_surf]
                .at[i_ts, ...]
                .set(finite_difference(i_ts, self.gamma_b[i_surf], dt, 0, order=1))
            )

    def calculate_unsteady_forcing(self, i_ts: int, ncs: ArrayList, rho: Array) -> None:
        r"""
        Calculate unsteady aerodynamic forcing for all surfaces at specified time step.
        :param i_ts: Timestep index
        :param ncs: Bound normal vectors for each surface, [n_surf][m, n, 3]
        :param rho: Fluid density
        """
        for i_surf in range(self.n_surf):
            self.f_unsteady[i_surf] = (
                self.f_unsteady[i_surf]
                .at[i_ts, ...]
                .set(
                    self._calculate_surf_unsteady_forcing(
                        i_ts, i_surf, ncs[i_surf], rho=rho
                    )
                )
            )

    def project_forcing_to_beam(
        self,
        i_ts: int,
        rmat: Array,
        x0_aero: ArrayList,
        dof_mapping: tuple[Array, ...],
        include_unsteady: bool,
    ) -> Array:
        r"""
        Project aerodynamic forcing at specified time step onto the beam grid.
        :param i_ts: Timestep index.
        :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
        :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
        :param dof_mapping: Tuple of arrays mapping aerodynamic grid points to beam nodes for each surface, [n_surf][zeta_n]
        :param include_unsteady: If true, include unsteady forcing in projection, otherwise only project steady forcing.
        :return: Steady and unsteady forcing projected onto the beam grid, [n_nodes, 6]
        """

        n_nodes = rmat.shape[0]
        result = jnp.zeros((n_nodes, 6))

        for i_surf in range(self.n_surf):
            # forcing for this surface
            this_force = self.f_steady[i_surf][i_ts, ...]  # [ zeta_m, zeta_n, 3]
            if include_unsteady:
                this_force += self.f_unsteady[i_surf][i_ts, ...]

            # rotate relative distances to get moment arms
            this_rmat = rmat[dof_mapping[i_surf], ...]  # [zeta_n, 3, 3]
            r_x0 = jnp.einsum(
                "ijk,lik->lij", this_rmat, x0_aero[i_surf]
            )  # relative distance [zeta_n, zeta_m, 3]

            result = result.at[dof_mapping[i_surf], :3].set(
                this_force.sum(axis=0)
            )  # forcing is sum along strip [zeta_n, 3]
            result = result.at[dof_mapping[i_surf], 3:].set(
                jnp.cross(r_x0, this_force).sum(axis=0)
            )  # moment is r x f summed along strip [zeta_n, 3]
        return result

    def _calculate_surf_unsteady_forcing(
        self, i_ts: int, i_surf: int, nc: Array, rho: Array
    ) -> Array:
        r"""
        Calculate unsteady aerodynamic forcing for a single surfaces at specified time step.
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :param nc: Bound normal vectors, [m, n, 3]
        :return: Unsteady aerodynamic forcing for surface at grid vertex, [zeta_m, zeta_n, 3]
        """
        return split_to_vertex(
            rho * self.gamma_b_dot[i_surf][i_ts, ..., None] * nc, (0, 1)
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "surf_b_names", "surf_w_names", "n_surf", "n_tstep"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "zeta_b",
            "zeta_b_dot",
            "zeta_w",
            "gamma_b",
            "gamma_b_dot",
            "gamma_w",
            "f_steady",
            "f_unsteady",
            "t",
        )


@_make_pytree
class StaticAero:
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
        i_ts: Optional[int],
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
        self.i_ts: Optional[int] = i_ts
        self.t: Array = t
        self.n_surf: int = n_surf

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        r"""
        Get snapshot for a single aerodynamic surface.
        :param i_surf: Index of aerodynamic surface
        :return: AeroSurfaceSnapshot for the specified surface
        """
        if i_surf < 0 or i_surf >= self.n_surf:
            raise IndexError("StaticAero index out of range")

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

    def project_forcing_to_beam(
        self,
        rmat: Array,
        x0_aero: ArrayList,
        dof_mapping: tuple[Array, ...],
        include_unsteady: bool,
    ) -> Array:
        r"""
        Project aerodynamic forcing at specified time step onto the beam grid.
        :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
        :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
        :param dof_mapping: Tuple of arrays mapping aerodynamic grid points to beam nodes for each surface, [n_surf][zeta_n]
        :param include_unsteady: If true, include unsteady forcing in projection, otherwise only project steady forcing.
        :return: Steady and unsteady forcing projected onto the beam grid, [n_nodes, 6]
        """

        n_nodes = rmat.shape[0]
        result = jnp.zeros((n_nodes, 6))

        for i_surf in range(self.n_surf):
            # forcing for this surface
            this_force = self.f_steady[i_surf]  # [ zeta_m, zeta_n, 3]
            if include_unsteady:
                this_force += self.f_unsteady[i_surf]

            # rotate relative distances to get moment arms
            this_rmat = rmat[dof_mapping[i_surf], ...]  # [zeta_n, 3, 3]
            r_x0 = jnp.einsum(
                "ijk,lik->lij", this_rmat, x0_aero[i_surf]
            )  # relative distance [zeta_n, zeta_m, 3]

            result = result.at[dof_mapping[i_surf], :3].set(
                this_force.sum(axis=0)
            )  # forcing is sum along strip [zeta_n, 3]
            result = result.at[dof_mapping[i_surf], 3:].set(
                jnp.cross(r_x0, this_force).sum(axis=0)
            )  # moment is r x f summed along strip [zeta_n, 3]
        return result

    def plot(
        self, directory: str | PathLike, plot_bound: bool = True, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot all aerodynamic surfaces in the snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        paths = []
        for i_surf in range(self.n_surf):
            paths.extend(
                self[i_surf].plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)
            )
        return paths

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "surf_b_names", "surf_w_names", "n_surf"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "zeta_b",
            "zeta_b_dot",
            "zeta_w",
            "gamma_b",
            "gamma_b_dot",
            "gamma_w",
            "f_steady",
            "f_unsteady",
            "t",
            "i_ts",
        )


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
        self, directory: str | PathLike, plot_bound: bool = True, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot aerodynamic surface in the snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_bound: If True, plot the bound surface.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
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


class AeroLinearResult:
    def __init__(
        self,
        u_t: Optional[InputUnflattened],
        x_t: Optional[StateUnflattened],
        y_t: Optional[OutputUnflattened],
        u_t_tot: Optional[InputUnflattened],
        x_t_tot: Optional[StateUnflattened],
        y_t_tot: Optional[OutputUnflattened],
        n_tstep: Optional[int],
        n_surf: int,
        t: Optional[Array],
        surf_b_names: list[str],
        surf_w_names: list[str],
    ) -> None:
        # system results, if simulated
        self.u_t: Optional[InputUnflattened] = u_t
        self.x_t: Optional[StateUnflattened] = x_t
        self.y_t: Optional[OutputUnflattened] = y_t
        self.u_t_tot: Optional[InputUnflattened] = u_t_tot
        self.x_t_tot: Optional[StateUnflattened] = x_t_tot
        self.y_t_tot: Optional[OutputUnflattened] = y_t_tot
        self.n_tstep: Optional[int] = n_tstep
        self.n_surf: int = n_surf
        self.t: Optional[Array] = t
        self.surf_b_names: list[str] = surf_b_names
        self.surf_w_names: list[str] = surf_w_names

    def plot(
        self,
        directory: str | PathLike,
        index: Optional[slice | Sequence[int] | int | Array] = None,
        plot_wake: bool = True,
    ) -> None:
        r"""
        Plot the aerodynamic grid at specified time steps.
        :param directory: Directory to save the plots to
        :param index: Index or slice of time steps to plot. If None, plot all time steps.
        :param plot_wake: If True, plot the wake grid
        """
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index,)
        elif index is None:
            index_ = jnp.arange(self.n_tstep)
        else:
            raise TypeError("index must be a slices, sequence of ints, or Array")

        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)

        paths: list[Sequence[Path]] = []
        for i_ts in index_:
            snapshot = self[i_ts]
            paths.append(snapshot.plot(directory, plot_wake=plot_wake))

        for i_surf in range(2 * self.n_surf):
            try:
                surf_paths = [paths[i][i_surf] for i in range(len(index_))]
                name = (self.surf_b_names + self.surf_w_names)[i_surf] + "_ts"
                write_pvd(directory, name, surf_paths, list(self.t[index_]))
            except IndexError:
                pass

    def __getitem__(self, i_ts: int) -> StaticAero:
        r"""
        Get snapshot of aerodynamic surface at a single time step
        :param i_ts: Timestep index
        :return: StaticAero at specified time step
        """

        if i_ts < 0 or i_ts >= self.n_tstep:
            raise IndexError("Timestep index out of range")

        # always exist
        zeta_b_tot = ArrayList(
            [self.u_t_tot.zeta_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        zeta_b_dot_tot = ArrayList(
            [
                self.u_t_tot.zeta_b_dot[i_surf][i_ts, ...]
                for i_surf in range(self.n_surf)
            ]
        )
        gamma_b_tot = ArrayList(
            [self.x_t_tot.gamma_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        gamma_w_tot = ArrayList(
            [self.x_t_tot.gamma_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        f_steady_tot = ArrayList(
            [self.y_t_tot.f_steady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )

        # optional
        gamma_b_dot_tot = (
            ArrayList(
                [
                    self.x_t_tot.gamma_b_dot[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.x_t_tot.gamma_b_dot is not None
            else None
        )
        zeta_w_tot = (
            ArrayList(
                [
                    self.x_t_tot.zeta_w[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.x_t_tot.zeta_w is not None
            else None
        )
        f_unsteady_tot = (
            ArrayList(
                [
                    self.y_t_tot.f_unsteady[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.y_t_tot.f_unsteady is not None
            else None
        )

        return StaticAero(
            zeta_b=zeta_b_tot,
            zeta_b_dot=zeta_b_dot_tot,
            zeta_w=zeta_w_tot,
            gamma_b=gamma_b_tot,
            gamma_b_dot=gamma_b_dot_tot,
            gamma_w=gamma_w_tot,
            f_steady=f_steady_tot,
            f_unsteady=f_unsteady_tot,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t[i_ts],
            n_surf=self.n_surf,
        )


@jax.tree_util.register_dataclass
@dataclass
class AeroStates:
    f_steady: Array
    f_unsteady: Array
    gamma_b: Array
    gamma_w: Array


@_make_pytree
class AeroDesignVariables:
    def __init__(
        self,
        x0_aero: Optional[ArrayList | Sequence[Array] | Array],
        flowfield: Optional[FlowField],
    ):
        self.x0_aero: Optional[ArrayList] = (
            ArrayList(x0_aero) if x0_aero is not None else None
        )
        self.flowfield: Optional[FlowField] = flowfield

        self.shapes: dict[str, Optional[tuple[int, ...]]] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def get_vars(self) -> dict[str, Optional[Array]]:
        return {
            "x0_aero": self.x0_aero,
            "flowfield": self.flowfield,
        }

    def get_shapes(self) -> dict[str, Optional[tuple[int, ...]]]:
        return {
            k: var.shape if var is not None else None
            for k, var in self.get_vars().items()
        }

    def make_index_mapping(self) -> tuple[dict[str, Optional[Array]], int]:
        mapping = {}
        cnt = 0
        for name, shape in self.shapes.items():
            if shape is not None:
                var_size = jnp.prod(jnp.array(shape))
                mapping[name] = jnp.arange(cnt, cnt + var_size).reshape(shape)
                cnt += var_size
            else:
                mapping[name] = None
        return mapping, cnt

    def ravel_jacobian(self, f_size: int, x_size: int) -> Array:
        arr = jnp.concatenate(
            [
                var.reshape(f_size, -1)
                for var in self.get_vars().values()
                if var is not None
            ],
            axis=1,
        )
        check_arr_shape(arr, (f_size, x_size), "Internal jacobian")
        return arr

    def ravel(self) -> Array:
        return jnp.concatenate(
            [var.ravel() for var in self.get_vars().values() if var is not None]
        )

    def reshape(self, *args: int) -> Array:
        return self.ravel().reshape(*args)

    def from_adjoint(
        self, f_shape: tuple[int, ...], df_dx: Array
    ) -> AeroDesignVariables:
        out_dict = {}
        for name in self.shapes.keys():
            if self.mapping[name] is not None:
                out_dict[name] = df_dx[:, self.mapping[name]].reshape(
                    *f_shape, *self.shapes[name]
                )
            else:
                out_dict[name] = None
        return AeroDesignVariables(**out_dict)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "shapes", "mapping"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "x0_aero", "flowfield"
