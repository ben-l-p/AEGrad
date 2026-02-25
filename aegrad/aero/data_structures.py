from __future__ import annotations

from typing import Sequence, Optional
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from jax import numpy as jnp
from jax import Array

from aegrad.print_utils import warn
from aegrad.algebra.base import finite_difference
from aegrad.algebra.array_utils import split_to_vertex
from algebra.array_utils import ArrayList
from plotting.aerogrid import plot_grid_to_vtk
from utils import _make_pytree


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
        dof_mapping: ArrayList,
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
        self.dof_mapping: ArrayList = dof_mapping
        self.horseshoe: bool = False

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
            horseshoe=False,
            dof_mapping=self.dof_mapping,
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
        return "surf_b_names", "surf_w_names", "n_surf", "n_tstep", "horseshoe"

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
        horseshoe: bool,
        dof_mapping: ArrayList,
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
        self.horseshoe: bool = horseshoe
        self.dof_mapping: ArrayList = dof_mapping

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
            horseshoe=self.horseshoe,
            dof_mapping=self.dof_mapping[i_surf],
        )

    def project_forcing_to_beam(
        self,
        rmat: Array,
        x0_aero: ArrayList,
        include_unsteady: bool,
    ) -> Array:
        r"""
        Project aerodynamic forcing at specified time step onto the beam grid.
        :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
        :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
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
            this_rmat = rmat[self.dof_mapping[i_surf], ...]  # [zeta_n, 3, 3]
            r_x0 = jnp.einsum(
                "ijk,lik->lij", this_rmat, x0_aero[i_surf]
            )  # relative distance [zeta_n, zeta_m, 3]

            result = result.at[self.dof_mapping[i_surf], :3].set(
                this_force.sum(axis=0)
            )  # forcing is sum along strip [zeta_n, 3]
            result = result.at[self.dof_mapping[i_surf], 3:].set(
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
        return "surf_b_names", "surf_w_names", "n_surf", "horseshoe", "dof_mapping"

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
        horseshoe: bool,
        dof_mapping: Array,
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
        self.horseshoe: bool = horseshoe
        self.dof_mapping: Array = dof_mapping

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
                plot_grid_to_vtk(
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
                    plot_grid_to_vtk(
                        self.zeta_w,
                        wake_filename,
                        self.i_ts,
                        cell_scalar_data={"gamma": self.gamma_w},
                    )
                )
        return paths
