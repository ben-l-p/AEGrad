from __future__ import annotations

from typing import Sequence, Optional
from dataclasses import dataclass
import os
from pathlib import Path

from jax import numpy as jnp
from jax import Array

from aero.aic import compute_v_ind
from aero.gradients.data_structures import AeroFullStates
from aero.utils import (
    KernelFunction,
    compute_c,
    compute_nc,
    calculate_steady_forcing, project_forcing_to_beam,
)
from print_utils import warn
from algebra.array_utils import split_to_vertex
from aero.flowfields import FlowField
from algebra.array_utils import ArrayList
from plotting.aerogrid import plot_grid_to_vtk
from plotting.pvd import write_pvd
from utils import _make_pytree, index_to_arr


@dataclass
class GridDiscretization:
    r"""
    Data class to hold grid discretisation parameters
    :param m: Number of panels in the chordwise direction
    :param n: Number of panels in the spanwise direction
    :param m_star: Number of wake panels in the chordwise direction
    """

    m: int
    n: int
    m_star: int


@_make_pytree
class DynamicAeroCase:
    r"""
    Class to hold time series of multiple aerodynamic surfaces.
    """

    def __init__(
            self,
            zeta_b: ArrayList,
            zeta_b_dot: ArrayList,
            zeta_w: Optional[ArrayList],
            c: Optional[ArrayList],
            nc: Optional[ArrayList],
            gamma_b: ArrayList,
            gamma_b_dot: Optional[ArrayList],
            gamma_w: ArrayList,
            f_steady: ArrayList,
            f_unsteady: Optional[ArrayList],
            kernels: Sequence[KernelFunction],
            mirror_point: Optional[Array],
            mirror_normal: Optional[Array],
            flowfield: FlowField,
            surf_b_names: Sequence[str],
            surf_w_names: Sequence[str],
            t: Array,
            i_ts: int | Array,
            dof_mapping: ArrayList,
            horseshoe: bool = False,
    ) -> None:
        r"""
        Time series of multiple aerodynamic surfaces
        :param zeta_b: Bound grid coordinates, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param zeta_b_dot: Bound grid velocities, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param zeta_w: Wake grid coordinates, [n_surf][n_ts, zeta_m_star, zeta_n, 3]
        :param gamma_b: Bound circulation strengths, [n_surf][n_ts, m, varphi]
        :param gamma_w: Wake circulation strengths, [n_surf][n_ts, m_star, varphi]
        :param f_steady: Steady force contributions, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param f_unsteady: Unsteady force contributions, [n_surf][n_ts, zeta_m, zeta_n, 3]
        :param surf_b_names: Names of bound surfaces, [n_surf]
        :param surf_w_names: Names of wake surfaces, [n_surf]
        :param t: Time array for the time series
        """
        self.zeta_b: ArrayList = zeta_b
        self.zeta_b_dot: ArrayList = zeta_b_dot
        self.zeta_w: Optional[ArrayList] = zeta_w
        self.c: Optional[ArrayList] = c
        self.nc: Optional[ArrayList] = nc
        self.gamma_b: ArrayList = gamma_b
        self.gamma_b_dot: Optional[ArrayList] = gamma_b_dot
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: Optional[ArrayList] = f_unsteady
        self.t: Array = t
        self.i_ts: Array | int = i_ts

        self.kernels: Sequence[KernelFunction] = kernels
        self.mirror_point: Optional[Array] = mirror_point
        self.mirror_normal: Optional[Array] = mirror_normal
        self.flowfield: FlowField = flowfield
        self.surf_b_names: Sequence[str] = surf_b_names
        self.surf_w_names: Sequence[str] = surf_w_names

        self.n_surf: int = len(surf_b_names)
        self.n_tstep: int = len(t)
        self.dof_mapping: ArrayList = dof_mapping
        self.horseshoe: bool = horseshoe

    @property
    def zeta_b(self) -> ArrayList:
        return self._zeta_b

    @zeta_b.setter
    def zeta_b(self, zeta_b_list: ArrayList) -> None:
        self._zeta_b = zeta_b_list

    @property
    def zeta_b_dot(self) -> ArrayList:
        return self._zeta_b_dot

    @zeta_b_dot.setter
    def zeta_b_dot(self, zeta_b_dot_list: ArrayList) -> None:
        self._zeta_b_dot = zeta_b_dot_list

    @property
    def zeta_w(self) -> ArrayList:
        if self._zeta_w is None: raise ValueError('zeta_w is not set')
        return self._zeta_w

    @zeta_w.setter
    def zeta_w(self, zeta_w_list: ArrayList) -> None:
        self._zeta_w = zeta_w_list

    @property
    def c(self) -> ArrayList:
        if self._c is None: raise ValueError('c is not set')
        return self._c

    @c.setter
    def c(self, c_list: ArrayList) -> None:
        self._c = c_list

    @property
    def nc(self) -> ArrayList:
        if self._nc is None: raise ValueError('nc is not set')
        return self._nc

    @nc.setter
    def nc(self, nc_list: ArrayList) -> None:
        self._nc = nc_list

    @property
    def gamma_b(self) -> ArrayList:
        return self._gamma_b

    @gamma_b.setter
    def gamma_b(self, gamma_b_list: ArrayList) -> None:
        self._gamma_b = gamma_b_list

    @property
    def gamma_b_dot(self) -> ArrayList:
        if self._gamma_b_dot is None: raise ValueError('gamma_b_dot is not set')
        return self._gamma_b_dot

    @gamma_b_dot.setter
    def gamma_b_dot(self, gamma_b_dot_list: ArrayList) -> None:
        self._gamma_b_dot = gamma_b_dot_list

    @property
    def gamma_w(self) -> ArrayList:
        return self._gamma_w

    @gamma_w.setter
    def gamma_w(self, gamma_w_list: ArrayList) -> None:
        self._gamma_w = gamma_w_list

    @property
    def f_steady(self) -> ArrayList:
        return self._f_steady

    @f_steady.setter
    def f_steady(self, f_steady_list: ArrayList) -> None:
        self._f_steady = f_steady_list

    @property
    def f_unsteady(self) -> ArrayList:
        if self._f_unsteady is None: raise ValueError('f_unsteady is not set')
        return self._f_unsteady

    @f_unsteady.setter
    def f_unsteady(self, f_unsteady_list: ArrayList) -> None:
        self._f_unsteady = f_unsteady_list

    @property
    def t(self) -> Array:
        return self._t

    @t.setter
    def t(self, t_arr: Array) -> None:
        self._t = t_arr

    @property
    def i_ts(self) -> Array | int:
        return self._i_ts

    @i_ts.setter
    def i_ts(self, i_ts_arr: Array | int) -> None:
        self._i_ts = i_ts_arr

    def get_full_states(self) -> AeroFullStates:
        return AeroFullStates(f_steady=self.f_steady, f_unsteady=self.f_unsteady, gamma_b=self.gamma_b,
                              gamma_w=self.gamma_w)

    def gamma_full(self, i_ts: int) -> ArrayList:
        r"""
        Obtain the full bound and wake gamma at a given timestep.
        :param i_ts: Time step index
        :return: Circulation strength, [2 * n_surf][m, varphi]
        """
        return ArrayList(
            [*self.gamma_b.index_all(i_ts, ...), *self.gamma_w.index_all(i_ts, ...)]
        )

    def zeta_full(self, i_ts: int) -> ArrayList:
        return ArrayList(
            [*self.zeta_b.index_all(i_ts, ...), *self.zeta_w.index_all(i_ts, ...)]
        )

    def set_arraylist_at_ts(self, attr: str, values: ArrayList, i_ts: int) -> None:
        """
        Sets self.attr with a given ArrayList of values at a given timestep.
        :param attr: Name of attribute in class
        :param values: Values to set
        :param i_ts: Time step index
        """
        arr = getattr(self, attr)
        for i_surf, val in enumerate(values):
            arr[i_surf] = arr[i_surf].at[i_ts, ...].set(val)

    def get_surf_snapshot(self, i_ts: int, i_surf: int) -> AeroSurfaceSnapshot:
        r"""
        Get initial_snapshot at a given timestep and surface.
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :return: StaticAero for the specified timestep
        """

        return AeroSurfaceSnapshot(
            zeta_b=self.zeta_b[i_surf][i_ts, ...],
            zeta_b_dot=self.zeta_b_dot[i_surf][i_ts, ...],
            zeta_w=self.zeta_w[i_surf][i_ts, ...],
            gamma_b=self.gamma_b[i_surf][i_ts, ...],
            gamma_b_dot=self.gamma_b_dot[i_surf][i_ts, ...],
            gamma_w=self.gamma_w[i_surf][i_ts, ...],
            f_steady=self.f_steady[i_surf][i_ts, ...],
            f_unsteady=self.f_unsteady[i_surf][i_ts, ...],
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=i_ts,
            t=self.t[i_ts],
            horseshoe=self.horseshoe,
            dof_mapping=self.dof_mapping[i_surf],
        )

    def __setitem__(
            self,
            i_ts: int,
            snapshot: DynamicAeroCase,
    ) -> None:
        if snapshot.n_tstep != 1:
            raise ValueError(
                "Snapshot must have n_tstep = 1 to set into DynamicAeroCase"
            )

        self._t = self._t.at[i_ts].set(snapshot.t[0])
        for i_surf in range(self.n_surf):
            self._zeta_b[i_surf] = (
                self._zeta_b[i_surf].at[i_ts, ...].set(snapshot.zeta_b[i_surf][0, ...])
            )
            if self._zeta_b_dot is not None and snapshot.zeta_b_dot is not None:
                self._zeta_b_dot[i_surf] = (
                    self._zeta_b_dot[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot.zeta_b_dot[i_surf][0, ...])
                )

            if self._zeta_w is not None and snapshot._zeta_w is not None:
                self._zeta_w[i_surf] = (
                    self._zeta_w[i_surf].at[i_ts, ...].set(snapshot._zeta_w[i_surf][0, ...])
                )
            self._gamma_b[i_surf] = (
                self._gamma_b[i_surf]
                .at[i_ts, ...]
                .set(snapshot.gamma_b[i_surf][0, ...])
            )
            if self._gamma_b_dot is not None and snapshot.gamma_b_dot is not None:
                self._gamma_b_dot[i_surf] = (
                    self._gamma_b_dot[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot.gamma_b_dot[i_surf][0, ...])
                )
            self._gamma_w[i_surf] = (
                self._gamma_w[i_surf]
                .at[i_ts, ...]
                .set(snapshot.gamma_w[i_surf][0, ...])
            )
            self._f_steady[i_surf] = (
                self._f_steady[i_surf]
                .at[i_ts, ...]
                .set(snapshot.f_steady[i_surf][0, ...])
            )
            if self._f_unsteady is not None and snapshot.f_unsteady is not None:
                self._f_unsteady[i_surf] = (
                    self._f_unsteady[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot.f_unsteady[i_surf][0, ...])
                )

    def plot(
            self, directory: os.PathLike, index: Optional[int | Sequence[int] | Array | slice] = None,
            plot_bound: bool = True,
            plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot all aerodynamic surfaces in the time series initial_snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param index: Index of timesteps to plot
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """

        index_ = index_to_arr(index=index, n_entries=self.n_tstep)

        pvd_paths = []
        for i_surf in range(self.n_surf):
            paths = []

            for i_ts in index_:
                paths.append(
                    self.get_surf_snapshot(i_ts=i_ts, i_surf=i_surf).plot(
                        directory, plot_bound=plot_bound, plot_wake=plot_wake
                    )
                )

            if plot_bound:
                bound_name = f"aero_dynamic_{self.surf_b_names[i_surf]}_ts"
                pvd_paths.append(write_pvd(directory=directory, name=bound_name, filedirs=list(zip(*paths))[0],
                                           times=list(self.t[index_])))

            if plot_wake:
                wake_name = f"aero_dynamic_{self.surf_w_names[i_surf]}_ts"
                pvd_paths.append(write_pvd(directory=directory, name=wake_name, filedirs=list(zip(*paths))[-1],
                                           times=list(self.t[index_])))
        return pvd_paths

    def get_c(self, i_ts: int) -> ArrayList:
        r"""
        Get collocation points for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of collocation points, [n_surf][m, varphi, 3]
        """
        if self._c is None: raise ValueError("No collocation points available")
        return self._c.index_all(i_ts, ...)

    def compute_c(self, i_ts: int) -> None:
        r"""
        Compute collocation points for all surfaces at specified time step and store in-place.
        :param i_ts: Timestep index
        """
        if self._c is None: raise ValueError("No collocation points available")
        c_list = compute_c(self._zeta_b.index_all(i_ts, ...))
        self.set_arraylist_at_ts("_c", c_list, i_ts)

    def get_n(self, i_ts: int) -> ArrayList:
        r"""
        Get varphi vectors for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of varphi vectors, [n_surf][m, varphi, 3]
        """
        if self._nc is None: raise ValueError("No normal vectors available")
        return self._nc.index_all(i_ts, ...)

    def compute_nc(self, i_ts: int) -> None:
        r"""
        Compute varphi vectors for all surfaces at specified time step and store in-place.
        :param i_ts: Timestep index
        """
        if self._nc is None: raise ValueError("No normal vectors available")
        nc_list = compute_nc(self._zeta_b.index_all(i_ts, ...))
        self.set_arraylist_at_ts("_nc", nc_list, i_ts)

    def set_gamma_w_static(self, i_ts: int) -> None:
        r"""
        Set wake circulation strengths in static solution to all match trailing edge strengths.
        :param i_ts: Timestep index
        """
        # Set wake panels equal to trailing edge gamma for each surface
        for i_surf in range(self.n_surf):
            val = self._gamma_b[i_surf][i_ts, [-1], :]
            self._gamma_w[i_surf] = self._gamma_w[i_surf].at[i_ts, ...].set(val)

    def calculate_steady_forcing(self, i_ts: int) -> None:
        r"""
        Calculate steady aerodynamic forcing for all surfaces at specified time step
        :param i_ts: Timestep index
        """

        f_steady = calculate_steady_forcing(
            zeta_bs=self.zeta_b.index_all(i_ts, ...),
            zeta_dot_bs=self.zeta_b_dot.index_all(i_ts, ...),
            gamma_bs=self.gamma_b.index_all(i_ts, ...),
            gamma_ws=self.gamma_w.index_all(i_ts, ...),
            rho=self.flowfield.rho,
            v_func=lambda x: self.get_v_tot(x=x, i_ts=i_ts),
            v_inputs=None,
        )

        for i_surf in range(self.n_surf):
            self._f_steady[i_surf] = (
                self._f_steady[i_surf].at[i_ts, ...].set(f_steady[i_surf])
            )

    def calculate_unsteady_forcing(
            self,
            i_ts: int,
    ) -> None:
        r"""
        Calculate unsteady aerodynamic forcing for all surfaces at specified time step.
        :param i_ts: Timestep index
        """
        if self._nc is None: raise ValueError("No normal vectors available")
        if self._f_unsteady is None: raise ValueError("No unsteady forcing available")
        for i_surf in range(self.n_surf):
            val = self._calculate_surf_unsteady_forcing(
                i_ts, i_surf, self._nc[i_surf][i_ts, ...], rho=self.flowfield.rho
            )
            self._f_unsteady[i_surf] = self._f_unsteady[i_surf].at[i_ts, ...].set(val)

    def project_forcing_to_beam(
            self,
            i_ts: int,
            rmat: Array,
            x0_aero: ArrayList,
            include_unsteady: bool,
    ) -> Array:
        r"""
        Project aerodynamic forcing at specified time step onto the beam grid. Returned forces are in the global frame.
        :param i_ts: Timestep index.
        :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
        :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
        :param include_unsteady: If true, include unsteady forcing in projection, otherwise only project steady forcing.
        :return: Steady and unsteady forcing projected onto the beam grid, [n_nodes, 6]
        """

        f_total = self._f_steady.index_all(i_ts, ...)
        if include_unsteady:
            f_total += self._f_unsteady.index_all(i_ts, ...)

        return project_forcing_to_beam(f_total=f_total, rmat=rmat, x0_aero=x0_aero, dof_mapping=self.dof_mapping)

    def _calculate_surf_unsteady_forcing(
            self, i_ts: int, i_surf: int, nc: Array, rho: Array
    ) -> Array:
        r"""
        Calculate unsteady aerodynamic forcing for a single surfaces at specified time step.
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :param nc: Bound varphi vectors, [m, varphi, 3]
        :return: Unsteady aerodynamic forcing for surface at grid vertex, [zeta_m, zeta_n, 3]
        """
        if self._gamma_b_dot is None: raise ValueError("No gamma_b_dot available")

        return split_to_vertex(
            rho * self._gamma_b_dot[i_surf][i_ts, ..., None] * nc, (0, 1)
        )

    def get_v_background[T: Array | ArrayList](self, i_ts: int, x_target: T) -> T:
        r"""
        Get background velocity at specified points and time step.
        :param i_ts: Timestep index
        :param x_target: Points to evaluate background velocity at, [][..., 3]
        :return: Background velocity at points, [][..., 3]
        """
        if isinstance(x_target, Array):
            return self.flowfield.vmap_call(x=x_target, t=self._t[i_ts])
        elif isinstance(x_target, ArrayList):
            return self.flowfield.surf_vmap_call(xs=x_target, t=self._t[i_ts])  # type: ignore
        else:
            raise NotImplementedError

    def get_v_ind[T: Array | ArrayList](self, i_ts: int, x_target: T) -> T:
        return compute_v_ind(
            cs=x_target,
            zetas=self.zeta_full(i_ts),
            gammas=self.gamma_full(i_ts),
            kernels=self.kernels,
            mirror_normal=self.mirror_normal,
            mirror_point=self.mirror_point,
        )

    def get_v_tot[T: Array | ArrayList](self, i_ts: int, x: T) -> T:
        r"""
        Obtain the total velocity at specified points and time step.
        :param i_ts: Timestep index
        :param x: ArrayList of points to evaluate total velocity at, [][..., 3]
        :return: Velocity at points, [][..., 3]
        """
        return self.get_v_ind(i_ts=i_ts, x_target=x) + self.get_v_background(
            i_ts=i_ts, x_target=x
        )

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        r"""
        Obtain a snapshot of the aerodynamic state at a given time step index.
        :param i_ts: Timestep index
        :return: AeroSnapshot object.
        """
        return AeroSnapshot(
            zeta_b=self._zeta_b.index_all(i_ts, ...),
            zeta_b_dot=self._zeta_b_dot.index_all(i_ts, ...),
            zeta_w=self._zeta_w.index_all(i_ts, ...) if self._zeta_w is not None else None,
            c=self._c.index_all(i_ts, ...) if self._c is not None else None,
            nc=self._nc.index_all(i_ts, ...) if self._nc is not None else None,
            gamma_b=self._gamma_b.index_all(i_ts, ...),
            gamma_b_dot=self._gamma_b_dot.index_all(i_ts, ...) if self._gamma_b_dot is not None else None,
            gamma_w=self._gamma_w.index_all(i_ts, ...),
            f_steady=self._f_steady.index_all(i_ts, ...),
            f_unsteady=self._f_unsteady.index_all(i_ts, ...) if self._f_unsteady is not None else None,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=self._t[i_ts],
            i_ts=i_ts,
            horseshoe=self.horseshoe,
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            dof_mapping=self.dof_mapping,
        )

    @classmethod
    def initialise(cls, initial_snapshot: AeroSnapshot, n_tstep: int) -> DynamicAeroCase:
        r"""
        Use a snapshot from a single timestep to create a solution object with many timesteps.
        :param initial_snapshot: Initial snapshot of the aerodynamic state.
        :param n_tstep: Number of timesteps.
        :return: New DynamicAeroCase object with n_tstep timesteps, with the initial case set to i_ts=0.
        """
        return initial_snapshot.to_dynamic(i_ts=0, n_tstep=n_tstep)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return (
            "surf_b_names",
            "surf_w_names",
            "n_surf",
            "_i_ts",
            "n_tstep",
            "horseshoe",
            "kernels",
            "mirror_point",
            "mirror_normal",
            "flowfield",
            "dof_mapping",
        )

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "_zeta_b",
            "_zeta_b_dot",
            "_zeta_w",
            "_c",
            "_nc",
            "_gamma_b",
            "_gamma_b_dot",
            "_gamma_w",
            "_f_steady",
            "_f_unsteady",
            "_t",
        )


@_make_pytree
class AeroSnapshot(DynamicAeroCase):
    r"""
    Class to hold initial_snapshot of multiple aerodynamic surfaces at a single time step.

    This class subclasses DynamicAeroCase but internally stores all arrays with a
    leading time dimension of length 1 so that it can reuse all of
    DynamicAeroCase's methods. When users request a single-surface initial_snapshot via
    indexing (initial_snapshot[i_surf]) they receive an AeroSurfaceSnapshot with the
    time dimension removed for convenience.
    """

    def __init__(
            self,
            zeta_b: ArrayList,
            zeta_b_dot: ArrayList,
            zeta_w: Optional[ArrayList],
            c: Optional[ArrayList],
            nc: Optional[ArrayList],
            gamma_b: ArrayList,
            gamma_b_dot: Optional[ArrayList],
            gamma_w: ArrayList,
            f_steady: ArrayList,
            f_unsteady: Optional[ArrayList],
            kernels: Sequence[KernelFunction],
            mirror_point: Optional[Array],
            mirror_normal: Optional[Array],
            flowfield: FlowField,
            surf_b_names: Sequence[str],
            surf_w_names: Sequence[str],
            t: float | Array,
            i_ts: int,
            dof_mapping: ArrayList,
            horseshoe: bool = False,
    ) -> None:
        r"""
        Create an AeroSnapshot by wrapping per-initial_snapshot arrays with a leading
        time dimension of size 1 so that DynamicAeroCase functions operate
        normally.
        """

        # call DynamicAeroCase initializer with expanded arrays
        super().__init__(
            zeta_b=zeta_b,
            zeta_b_dot=zeta_b_dot,
            zeta_w=zeta_w,
            c=c,
            nc=nc,
            gamma_b=gamma_b,
            gamma_b_dot=gamma_b_dot,
            gamma_w=gamma_w,
            f_steady=f_steady,
            f_unsteady=f_unsteady,
            kernels=kernels,
            mirror_point=mirror_point,
            mirror_normal=mirror_normal,
            flowfield=flowfield,
            surf_b_names=surf_b_names,
            surf_w_names=surf_w_names,
            t=jnp.atleast_1d(t),
            i_ts=i_ts,
            dof_mapping=dof_mapping,
            horseshoe=horseshoe,
        )

    @property
    def zeta_b(self) -> ArrayList:
        return self._zeta_b.index_all(0, ...)

    @zeta_b.setter
    def zeta_b(self, value: ArrayList) -> None:
        self._zeta_b = value.index_all(None, ...)

    @property
    def zeta_b_dot(self) -> ArrayList:
        return self._zeta_b_dot.index_all(0, ...)

    @zeta_b_dot.setter
    def zeta_b_dot(self, value: ArrayList) -> None:
        self._zeta_b_dot = value.index_all(None, ...)

    @property
    def zeta_w(self) -> ArrayList:
        if self._zeta_w is None: raise ValueError("zeta_w is None")
        return self._zeta_w.index_all(0, ...)

    @zeta_w.setter
    def zeta_w(self, value: ArrayList) -> None:
        self._zeta_w = value.index_all(None, ...)

    @property
    def c(self) -> ArrayList:
        if self._c is None: raise ValueError("c is None")
        return self._c.index_all(0, ...)

    @c.setter
    def c(self, value: Optional[ArrayList]) -> None:
        if value is not None:
            self._c = value.index_all(None, ...)

    @property
    def nc(self) -> ArrayList:
        if self._nc is None: raise ValueError("nc is None")
        return self._nc.index_all(0, ...)

    @nc.setter
    def nc(self, value: Optional[ArrayList]) -> None:
        if value is not None:
            self._nc = value.index_all(None, ...)

    @property
    def gamma_b(self) -> ArrayList:
        return self._gamma_b.index_all(0, ...)

    @gamma_b.setter
    def gamma_b(self, value: ArrayList) -> None:
        self._gamma_b = value.index_all(None, ...)

    @property
    def gamma_b_dot(self) -> ArrayList:
        if self._gamma_b_dot is None: raise ValueError("gamma_b_dot is None")
        return self._gamma_b_dot.index_all(0, ...)

    @gamma_b_dot.setter
    def gamma_b_dot(self, value: ArrayList) -> None:
        self._gamma_b_dot = value.index_all(None, ...)

    @property
    def gamma_w(self) -> ArrayList:
        return self._gamma_w.index_all(0, ...)

    @gamma_w.setter
    def gamma_w(self, value: ArrayList) -> None:
        self._gamma_w = value.index_all(None, ...)

    @property
    def f_steady(self) -> ArrayList:
        return self._f_steady.index_all(0, ...)

    @f_steady.setter
    def f_steady(self, value: ArrayList) -> None:
        self._f_steady = value.index_all(None, ...)

    @property
    def f_unsteady(self) -> ArrayList:
        if self._f_unsteady is None: raise ValueError("f_unsteady is None")
        return self._f_unsteady.index_all(0, ...)

    @f_unsteady.setter
    def f_unsteady(self, value: ArrayList) -> None:
        self._f_unsteady = value.index_all(None, ...)

    @property
    def t(self) -> Array:
        return self._t[0]

    @t.setter
    def t(self, t_val: Array) -> None:
        self._t = t_val

    @property
    def i_ts(self) -> int:
        return self._i_ts

    @i_ts.setter
    def i_ts(self, i_ts_val: int) -> None:
        self._i_ts = i_ts_val

    def to_dynamic(self, i_ts: int, n_tstep: int) -> DynamicAeroCase:
        """
        Expand this single-time initial_snapshot into a DynamicAeroCase with n_tstep
        timesteps, placing the current initial_snapshot at index i_ts (similar to the
        prior implementation).
        """

        def _expand_to_dyn(arr_list: ArrayList) -> ArrayList:
            out = []
            for a in arr_list:
                arr = jnp.zeros((n_tstep, *a.shape)).at[i_ts, ...].set(a)
                out.append(arr)
            return ArrayList(out)

        return DynamicAeroCase(
            zeta_b=_expand_to_dyn(self.zeta_b),
            zeta_b_dot=_expand_to_dyn(self.zeta_b_dot),
            zeta_w=_expand_to_dyn(self.zeta_w),
            c=_expand_to_dyn(self.c),
            nc=_expand_to_dyn(self.nc),
            gamma_b=_expand_to_dyn(self.gamma_b),
            gamma_b_dot=_expand_to_dyn(self.gamma_b_dot),
            gamma_w=_expand_to_dyn(self.gamma_w),
            f_steady=_expand_to_dyn(self.f_steady),
            f_unsteady=_expand_to_dyn(self.f_unsteady),
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=jnp.zeros(n_tstep).at[i_ts].set(self.t),
            i_ts=i_ts,
            dof_mapping=self.dof_mapping,
            horseshoe=self.horseshoe,
        )

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        """
        Return a single-surface initial_snapshot with the time dimension removed for
        convenience. Uses the stored i_ts index to pick the single time slice.
        """
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

    def plot(
            self,
            directory: os.PathLike | str,
            _=None,
            plot_bound: bool = True,
            plot_wake: bool = True,
    ) -> Sequence[Path]:
        """
        Plot all aerodynamic surfaces in this single-time initial_snapshot to VTU files.
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i_surf in range(self.n_surf):
            paths.extend(
                self[i_surf].plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)
            )

        return paths


class AeroSurfaceSnapshot:
    r"""
    Data class to hold initial_snapshot of a single aerodynamic surface at a single time step.
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
        :param gamma_b: Bound circulation strengths, [m, varphi]
        :param gamma_w: Wake circulation strengths, [m_star, varphi]
        :param f_steady: Steady force contributions, [zeta_m, zeta_n, 3]
        :param f_unsteady: Unsteady force contributions, [zeta_m, zeta_n, 3]
        :param surf_b_name: Names of bound surface
        :param surf_w_name: Names of wake surface
        :param i_ts: Time step index
        :param t: Time at this initial_snapshot
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
            self,
            directory: str | os.PathLike,
            plot_bound: bool = True,
            plot_wake: bool = True,
    ) -> Sequence[Path]:
        r"""
        Plot aerodynamic surface in the initial_snapshot to VTU files.
        :param directory: Directory to save VTU files.
        :param plot_bound: If True, plot the bound surface.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved VTU files.
        """

        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
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
