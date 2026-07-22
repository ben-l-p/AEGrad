from __future__ import annotations

from typing import Sequence, Optional
from dataclasses import dataclass
import os
from pathlib import Path

from jax import numpy as jnp
from jax import Array

from aegrad.aero.aic import compute_v_ind
from aegrad.aero.gradients.data_structures import AeroStates
from aegrad.aero.utils import (
    KernelFunction,
    project_forcing_to_beam,
)

from aegrad.utils.print_utils import warn
from aegrad.aero.flowfields import FlowField
from aegrad.algebra.array_utils import ArrayList
from aegrad.plotting.aerogrid import plot_grid_to_vtk
from aegrad.plotting.pvd import write_pvd
from aegrad.utils.utils import make_pytree, index_to_arr


@dataclass
class GridDiscretization:
    r"""
    Data class to hold discretisation parameters for each aerodynamic grid.
    :param m: Number of panels in the chordwise direction.
    :param n: Number of panels in the spanwise direction.
    :param m_star: Number of wake panels in the chordwise direction.
    """

    m: int
    n: int
    m_star: int


@make_pytree
class DynamicAeroCase:
    r"""
    Contains a solution for a dynamic time series of multiple aerodynamic surfaces.
    """

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: Optional[ArrayList],
        c: Optional[ArrayList],
        n: Optional[ArrayList],
        gamma_b: ArrayList,
        gamma_b_dot: Optional[ArrayList],
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: Optional[ArrayList],
        cs_ang: dict[str, Array],
        cs_vel: dict[str, Array],
        kernels: Sequence[KernelFunction],
        mirror_point: Optional[Array],
        mirror_normal: Optional[Array],
        flowfield: FlowField,
        surf_b_names: Sequence[str],
        surf_w_names: Sequence[str],
        t: Array,
        i_ts: Array,
        dof_mapping: ArrayList,
        static_horseshoe: bool,
        free_wake: bool,
        gamma_dot_relaxation: float,
        batch_size: Optional[int],
    ) -> None:
        r"""
        :param zeta_b: Bound grid coordinates, [n_surf][n_tstep, zeta_m, zeta_n, 3].
        :param zeta_b_dot: Bound grid velocities, [n_surf][n_tstep, zeta_m, zeta_n, 3].
        :param zeta_w: Wake grid coordinates, [n_surf][n_tstep, zeta_m_star, zeta_n, 3].
        :param c: Bound collocation points, [n_surf][n_tstep, m, n, 3].
        :param n: Bound grid normals, [n_surf][n_tstep, m, n, 3].
        :param gamma_b: Bound circulation strengths, [n_surf][n_tstep, m, n].
        :param gamma_b_dot:  Bound circulation time derivatives, [n_surf][n_tstep, m, n].
        :param gamma_w: Wake circulation strengths, [n_surf][n_tstep, m_star, n].
        :param f_steady: Steady force contributions, [n_surf][n_tstep, zeta_m, zeta_n, 3].
        :param f_unsteady: Unsteady force contributions, [n_surf][n_tstep, zeta_m, zeta_n, 3].
        :param cs_ang: Control surface angle time histories, {name: [n_tstep]}.
        :param cs_vel: Control surface velocity time history, {name: [n_tstep]}.
        :param kernels: Kernel functions for both bound and wake source grids.
        :param mirror_point: Point on mirror plane, [3] or None.
        :param mirror_normal: Normal on mirror plane, [3] or None.
        :param flowfield: FlowField object to obtain background velocity and density.
        :param surf_b_names: Names of bound surfaces, [n_surf].
        :param surf_w_names: Names of wake surfaces, [n_surf].
        :param t: Time array for the time series, [n_tstep].
        :param i_ts: Time series indices, [n_tstep].
        :param dof_mapping: Array for mapping the aerodynamic grid onto the beam degrees of freedom, [n_surf][zeta_n].
        :param static_horseshoe: If true, a horseshoe formulation was used to obtain the initial static solution.
        :param free_wake: Flag for if a free-wake formulation was used for solution.
        :param gamma_dot_relaxation: Circulation time derivative filtering parameter.
        :param batch_size: Batch size used for AIC vectorisation.
        """
        self.zeta_b: ArrayList = zeta_b
        self.zeta_b_dot: ArrayList = zeta_b_dot
        self.zeta_w: Optional[ArrayList] = zeta_w
        self.c: Optional[ArrayList] = c
        self.nc: Optional[ArrayList] = n
        self.gamma_b: ArrayList = gamma_b
        self.gamma_b_dot: Optional[ArrayList] = gamma_b_dot
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: Optional[ArrayList] = f_unsteady
        self.cs_ang: dict[str, Array] = cs_ang
        self.cs_vel: dict[str, Array] = cs_vel
        self.t: Array = t
        self.i_ts: Array = i_ts

        self.kernels: Sequence[KernelFunction] = kernels
        self.mirror_point: Optional[Array] = mirror_point
        self.mirror_normal: Optional[Array] = mirror_normal
        self.flowfield: FlowField = flowfield
        self.surf_b_names: Sequence[str] = surf_b_names
        self.surf_w_names: Sequence[str] = surf_w_names

        self.n_surf: int = len(surf_b_names)
        self.n_tstep: int = len(t)
        self.dof_mapping: ArrayList = dof_mapping

        # settings
        self.static_horseshoe: bool = static_horseshoe
        self.free_wake: bool = free_wake
        self.gamma_dot_relaxation: float = gamma_dot_relaxation
        self.batch_size: Optional[int] = batch_size

    # we use properties here to allow for subclasses where there are only a single time step, and so the output data
    # is preferred to have different dimensionality.

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
        if self._zeta_w is None:
            raise ValueError("zeta_w is not set")
        return self._zeta_w

    @zeta_w.setter
    def zeta_w(self, zeta_w_list: ArrayList) -> None:
        self._zeta_w = zeta_w_list

    @property
    def c(self) -> ArrayList:
        if self._c is None:
            raise ValueError("c is not set")
        return self._c

    @c.setter
    def c(self, c_list: ArrayList) -> None:
        self._c = c_list

    @property
    def nc(self) -> ArrayList:
        if self._nc is None:
            raise ValueError("n is not set")
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
        if self._gamma_b_dot is None:
            raise ValueError("gamma_b_dot is not set")
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
        if self._f_unsteady is None:
            raise ValueError("f_unsteady is not set")
        return self._f_unsteady

    @f_unsteady.setter
    def f_unsteady(self, f_unsteady_list: ArrayList) -> None:
        self._f_unsteady = f_unsteady_list

    @property
    def cs_ang(self) -> dict[str, Array]:
        return self._cs_ang

    @cs_ang.setter
    def cs_ang(self, value: dict[str, Array]) -> None:
        self._cs_ang = value

    @property
    def cs_vel(self) -> dict[str, Array]:
        return self._cs_vel

    @cs_vel.setter
    def cs_vel(self, value: dict[str, Array]) -> None:
        self._cs_vel = value

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

    def get_states(self, i_ts: int | Array) -> AeroStates:
        r"""
        Obtain the aerodynamic state at a given timestep, used in the adjoint solution.
        :param i_ts: Time step index.
        :return: Aero states.
        """
        return AeroStates(
            gamma_b=self.gamma_b.index_all(i_ts, ...),
            gamma_w=self.gamma_w.index_all(i_ts, ...),
            gamma_b_dot=self.gamma_b_dot.index_all(i_ts, ...),
            zeta_w=self.zeta_w.index_all(i_ts, ...),
        )

    def gamma_full(self, i_ts: int) -> ArrayList:
        r"""
        Obtain the full bound and wake circulation strengths at a given time step.
        :param i_ts: Time step index.
        :return: Circulation strength, [2 * n_surf][m | m_star, n].
        """
        return ArrayList(
            [*self.gamma_b.index_all(i_ts, ...), *self.gamma_w.index_all(i_ts, ...)]
        )

    def zeta_full(self, i_ts: int) -> ArrayList:
        r"""
        Obtain the full bound and wake grids at a given time step.
        :param i_ts: Time step index.
        :return: Grids, [2 * n_surf][zeta_m | zeta_m_star, zeta_n, 3].
        """
        return ArrayList(
            [*self.zeta_b.index_all(i_ts, ...), *self.zeta_w.index_all(i_ts, ...)]
        )

    def set_arraylist_at_ts(self, attr: str, values: ArrayList, i_ts: int) -> None:
        """
        Sets a given attribute with an ArrayList of values at a given timestep. This prevents needing a separate method
        for setting each attribute.
        :param attr: Name of attribute in class to set.
        :param values: ArrayList of values to set.
        :param i_ts: Time step index.
        """
        arr = getattr(self, attr)
        for i_surf, val in enumerate(values):
            arr[i_surf] = arr[i_surf].at[i_ts, ...].set(val)

    def get_surf_snapshot(self, i_ts: int, i_surf: int) -> AeroSurfaceSnapshot:
        r"""
        Get aerodynamic solution for a given timestep and surface.
        :param i_ts: Timestep index.
        :param i_surf: Surface index.
        :return: Data for aerodynamic solution at a given timestep and surface.
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
            cs_ang={k: v[i_ts, ...] for k, v in self.cs_ang.items()},
            cs_vel={k: v[i_ts, ...] for k, v in self.cs_vel.items()},
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=i_ts,
            t=self.t[i_ts],
            static_horseshoe=self.static_horseshoe,
            dof_mapping=self.dof_mapping[i_surf],
        )

    def __setitem__(
        self,
        i_ts: int,
        snapshot: DynamicAeroCase,
    ) -> None:
        r"""
        Sets the data at a given time step from a snapshot.
        :param i_ts: Time step index.
        :param snapshot: Snapshot of data to set at the given time step.
        """
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
                    self._zeta_w[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot._zeta_w[i_surf][0, ...])
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
        self,
        directory: os.PathLike,
        plot_bound: bool = True,
        plot_wake: bool = True,
        index: Optional[int | Sequence[int] | Array | slice] = None,
    ) -> Sequence[Path]:
        r"""
        Plot all aerodynamic surfaces in the time series to VTU files, with a corresponding PVD file for each surface.
        :param directory: Directory to save files.
        :param index: Index of timesteps to plot. If none, the full solution is saved.
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :return: Sequence of paths to the saved PVD files.
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
                pvd_paths.append(
                    write_pvd(
                        directory=directory,
                        name=bound_name,
                        file_dirs=list(zip(*paths))[0],
                        times=list(self.t[index_]),
                    )
                )

            if plot_wake:
                wake_name = f"aero_dynamic_{self.surf_w_names[i_surf]}_ts"
                pvd_paths.append(
                    write_pvd(
                        directory=directory,
                        name=wake_name,
                        file_dirs=list(zip(*paths))[-1],
                        times=list(self.t[index_]),
                    )
                )
        return pvd_paths

    def project_forcing_to_beam(
        self,
        i_ts: int,
        rmat: Array,
        x0_aero: ArrayList,
        include_unsteady: bool,
    ) -> Array:
        r"""
        Project aerodynamic forcing at specified time step onto the beam grid. Returned forces are in the global frame.
        :param i_ts: Time step index.
        :param rmat: Rotation matrix for each node relative to reference, [n_nodes, 3, 3].
        :param x0_aero: Reference coordinates for aerodynamic grid, [n_surf][zeta_m, zeta_n, 3].
        :param include_unsteady: If true, include unsteady forcing in projection, otherwise only project steady forcing.
        :return: Steady and unsteady forcing projected onto the beam grid in local frame, [n_nodes, 6].
        """

        f_total = self._f_steady.index_all(i_ts, ...)
        if include_unsteady:
            f_total += self._f_unsteady.index_all(i_ts, ...)

        return project_forcing_to_beam(
            f_total=f_total, rmat=rmat, x0_aero=x0_aero, dof_mapping=self.dof_mapping
        )

    def get_v_background[T: Array | ArrayList](self, i_ts: int, x_target: T) -> T:
        r"""
        Get background velocity at specified points and time step.
        :param i_ts: Time step index.
        :param x_target: Points to evaluate background velocity at, [..., 3] or [][..., 3].
        :return: Background velocity at points, [..., 3] or [][..., 3].
        """
        if isinstance(x_target, Array):
            return self.flowfield.vmap_call(x=x_target, t=self._t[i_ts])
        elif isinstance(x_target, ArrayList):
            return self.flowfield.surf_vmap_call(xs=x_target, t=self._t[i_ts])  # type: ignore
        else:
            raise NotImplementedError

    def get_v_ind[T: Array | ArrayList](self, i_ts: int, x_target: T) -> T:
        r"""
        Get induced velocity at specified points and time step.
        :param i_ts: Time step index.
        :param x_target: Points to evaluate induced velocity at, [..., 3] or [][..., 3].
        :return: Induced velocity at points, [..., 3] or [][..., 3].
        """
        return compute_v_ind(
            cs=x_target,
            zetas=self.zeta_full(i_ts),
            gammas=self.gamma_full(i_ts),
            kernels=self.kernels,
            mirror_normal=self.mirror_normal,
            mirror_point=self.mirror_point,
            batch_size=self.batch_size,
        )

    def get_v_tot[T: Array | ArrayList](self, i_ts: int, x_target: T) -> T:
        r"""
        Obtain the total velocity at specified points and time step.
        :param i_ts: Time step index.
        :param x_target: Points to evaluate induced velocity at, [..., 3] or [][..., 3].
        :return: Induced velocity at points, [..., 3] or [][..., 3].
        """
        return self.get_v_ind(i_ts=i_ts, x_target=x_target) + self.get_v_background(
            i_ts=i_ts, x_target=x_target
        )

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        r"""
        Obtain a snapshot of the aerodynamic state at a given time step index.
        :param i_ts: Time step index.
        :return: Solution for specified time step index.
        """
        return AeroSnapshot(
            zeta_b=self._zeta_b.index_all(i_ts, ...),
            zeta_b_dot=self._zeta_b_dot.index_all(i_ts, ...),
            zeta_w=self._zeta_w.index_all(i_ts, ...)
            if self._zeta_w is not None
            else None,
            c=self._c.index_all(i_ts, ...) if self._c is not None else None,
            n=self._nc.index_all(i_ts, ...) if self._nc is not None else None,
            gamma_b=self._gamma_b.index_all(i_ts, ...),
            gamma_b_dot=self._gamma_b_dot.index_all(i_ts, ...)
            if self._gamma_b_dot is not None
            else None,
            gamma_w=self._gamma_w.index_all(i_ts, ...),
            f_steady=self._f_steady.index_all(i_ts, ...),
            f_unsteady=self._f_unsteady.index_all(i_ts, ...)
            if self._f_unsteady is not None
            else None,
            cs_ang={k: jnp.atleast_1d(v)[i_ts, ...] for k, v in self.cs_ang.items()},
            cs_vel={k: jnp.atleast_1d(v)[i_ts, ...] for k, v in self.cs_vel.items()},
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=self._t[i_ts],
            i_ts=i_ts,
            static_horseshoe=self.static_horseshoe,
            free_wake=self.free_wake,
            gamma_dot_relaxation=self.gamma_dot_relaxation,
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            dof_mapping=self.dof_mapping,
            batch_size=self.batch_size,
        )

    @classmethod
    def initialise(
        cls, initial_snapshot: AeroSnapshot, n_tstep: int
    ) -> DynamicAeroCase:
        r"""
        Use a snapshot from a single timestep to create a solution object with many timesteps.
        :param initial_snapshot: Initial snapshot of the aerodynamic state.
        :param n_tstep: Number of timesteps.
        :return: New instance with n_tstep timesteps, with the initial case set at i_ts=0.
        """
        return initial_snapshot.to_dynamic(i_ts=0, n_tstep=n_tstep)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return (
            "surf_b_names",
            "surf_w_names",
            "n_surf",
            "n_tstep",
            "static_horseshoe",
            "gamma_dot_relaxation",
            "free_wake",
            "kernels",
            "batch_size",
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
            "_cs_ang",
            "_cs_vel",
            "_i_ts",
            "_t",
            "mirror_point",
            "mirror_normal",
            "flowfield",
            "dof_mapping",
        )


@make_pytree
class AeroSnapshot(DynamicAeroCase):
    r"""
    Class to hold initial_snapshot of multiple aerodynamic surfaces at a single time step.
    """

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: Optional[ArrayList],
        c: Optional[ArrayList],
        n: Optional[ArrayList],
        gamma_b: ArrayList,
        gamma_b_dot: Optional[ArrayList],
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: Optional[ArrayList],
        cs_ang: dict[str, Array],
        cs_vel: dict[str, Array],
        kernels: Sequence[KernelFunction],
        mirror_point: Optional[Array],
        mirror_normal: Optional[Array],
        flowfield: FlowField,
        surf_b_names: Sequence[str],
        surf_w_names: Sequence[str],
        t: float | Array,
        i_ts: int,
        dof_mapping: ArrayList,
        static_horseshoe: bool,
        free_wake: bool,
        gamma_dot_relaxation: float,
        batch_size: Optional[int],
    ) -> None:
        r"""
        :param zeta_b: Bound grid coordinates, [n_surf][zeta_m, zeta_n, 3].
        :param zeta_b_dot: Bound grid velocities, [n_surf][zeta_m, zeta_n, 3].
        :param zeta_w: Wake grid coordinates, [n_surf][zeta_m_star, zeta_n, 3].
        :param c: Bound collocation points, [n_surf][m, n, 3].
        :param n: Bound grid normals, [n_surf][m, n, 3].
        :param gamma_b: Bound circulation strengths, [n_surf][m, n].
        :param gamma_b_dot:  Bound circulation time derivatives, [n_surf][m, n].
        :param gamma_w: Wake circulation strengths, [n_surf][m_star, n].
        :param f_steady: Steady force contributions, [n_surf][zeta_m, zeta_n, 3].
        :param f_unsteady: Unsteady force contributions, [n_surf][zeta_m, zeta_n, 3].
        :param cs_ang: Control surface angle time histories, {name: []}.
        :param cs_vel: Control surface velocity time history, {name: []}.
        :param kernels: Kernel functions for both bound and wake source grids.
        :param mirror_point: Point on mirror plane, [3] or None.
        :param mirror_normal: Normal on mirror plane, [3] or None.
        :param flowfield: FlowField object to obtain background velocity and density.
        :param surf_b_names: Names of bound surfaces, [n_surf].
        :param surf_w_names: Names of wake surfaces, [n_surf].
        :param t: Time at the current snapshot, [].
        :param i_ts: Time series index in full solution, [].
        :param dof_mapping: Array for mapping the aerodynamic grid onto the beam degrees of freedom, [n_surf][zeta_n].
        :param static_horseshoe: If true, a horseshoe formulation was used to obtain the initial static solution.
        :param free_wake: Flag for if a free-wake formulation was used for solution.
        :param gamma_dot_relaxation: Circulation time derivative filtering parameter.
        :param batch_size: Batch size used for AIC vectorisation.
        """

        # call DynamicAeroCase initializer with expanded arrays
        super().__init__(
            zeta_b=zeta_b,
            zeta_b_dot=zeta_b_dot,
            zeta_w=zeta_w,
            c=c,
            n=n,
            gamma_b=gamma_b,
            gamma_b_dot=gamma_b_dot,
            gamma_w=gamma_w,
            f_steady=f_steady,
            f_unsteady=f_unsteady,
            cs_ang=cs_ang,
            cs_vel=cs_vel,
            kernels=kernels,
            mirror_point=mirror_point,
            mirror_normal=mirror_normal,
            flowfield=flowfield,
            surf_b_names=surf_b_names,
            surf_w_names=surf_w_names,
            t=jnp.atleast_1d(t),
            i_ts=jnp.atleast_1d(i_ts),
            dof_mapping=dof_mapping,
            static_horseshoe=static_horseshoe,
            free_wake=free_wake,
            gamma_dot_relaxation=gamma_dot_relaxation,
            batch_size=batch_size,
        )

    # redefine properties to allow internal representation (all variables with leading underscore) to have leading
    # dimension of 1.
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
        if self._zeta_w is None:
            raise ValueError("zeta_w is None")
        return self._zeta_w.index_all(0, ...)

    @zeta_w.setter
    def zeta_w(self, value: Optional[ArrayList]) -> None:
        if value is not None:
            self._zeta_w = value.index_all(None, ...)

    @property
    def c(self) -> ArrayList:
        if self._c is None:
            raise ValueError("c is None")
        return self._c.index_all(0, ...)

    @c.setter
    def c(self, value: Optional[ArrayList]) -> None:
        if value is not None:
            self._c = value.index_all(None, ...)

    @property
    def nc(self) -> ArrayList:
        if self._nc is None:
            raise ValueError("n is None")
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
        if self._gamma_b_dot is None:
            raise ValueError("gamma_b_dot is None")
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
        if self._f_unsteady is None:
            raise ValueError("f_unsteady is None")
        return self._f_unsteady.index_all(0, ...)

    @f_unsteady.setter
    def f_unsteady(self, value: ArrayList) -> None:
        self._f_unsteady = value.index_all(None, ...)

    @property
    def cs_ang(self) -> dict[str, Array]:
        return {k: v[0] for k, v in self._cs_ang.items()}

    @cs_ang.setter
    def cs_ang(self, value: dict[str, Array]) -> None:
        self._cs_ang = {k: v[None, ...] for k, v in value.items()}

    @property
    def cs_vel(self) -> dict[str, Array]:
        return {k: v[0] for k, v in self._cs_vel.items()}

    @cs_vel.setter
    def cs_vel(self, value: dict[str, Array]) -> None:
        self._cs_vel = {k: v[None, ...] for k, v in value.items()}

    @property
    def t(self) -> Array:
        return self._t[0]

    @t.setter
    def t(self, t_val: Array) -> None:
        self._t = t_val

    @property
    def i_ts(self) -> int:
        return int(self._i_ts[0])

    @i_ts.setter
    def i_ts(self, i_ts_val: int) -> None:
        self._i_ts = jnp.atleast_1d(i_ts_val)

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
            n=_expand_to_dyn(self.nc),
            gamma_b=_expand_to_dyn(self.gamma_b),
            gamma_b_dot=_expand_to_dyn(self.gamma_b_dot),
            gamma_w=_expand_to_dyn(self.gamma_w),
            f_steady=_expand_to_dyn(self.f_steady),
            f_unsteady=_expand_to_dyn(self.f_unsteady),
            cs_ang={k: jnp.full(n_tstep, v) for k, v in self.cs_ang.items()},
            cs_vel={k: jnp.full(n_tstep, v) for k, v in self.cs_vel.items()},
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=jnp.zeros(n_tstep).at[i_ts].set(self.t),
            i_ts=jnp.atleast_1d(i_ts),
            dof_mapping=self.dof_mapping,
            static_horseshoe=self.static_horseshoe,
            free_wake=self.free_wake,
            gamma_dot_relaxation=self.gamma_dot_relaxation,
            batch_size=self.batch_size,
        )

    def __getitem__(self, i_surf: int) -> AeroSurfaceSnapshot:
        """
        Return data for a single surface at the given time snapshot.
        :param i_surf: Aerodynamic surface index.
        :return: AeroSurfaceSnapshot object.
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
            cs_ang=self.cs_ang,
            cs_vel=self.cs_vel,
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=self.i_ts,
            t=self.t,
            static_horseshoe=self.static_horseshoe,
            dof_mapping=self.dof_mapping[i_surf],
        )

    def plot(
        self,
        directory: os.PathLike | str,
        plot_bound: bool = True,
        plot_wake: bool = True,
        _=None,
    ) -> Sequence[Path]:
        r"""
        Plot aerodynamic surfaces in the time snapshot to VTU files.
        :param directory: Directory to save files.
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :param _: Unused inherited argument.
        :return: Sequence of paths to the saved PVD files.
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
    Data class to hold solution for a single aerodynamic surface at a single time step.
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
        cs_ang: dict[str, Array],
        cs_vel: dict[str, Array],
        surf_b_name: str,
        surf_w_name: str,
        i_ts: int,
        t: Array,
        static_horseshoe: bool,
        dof_mapping: Array,
    ) -> None:
        r"""
        Data structure to hold the solution for a single aerodynamic surface at a single time step.
        :param zeta_b: Bound grid coordinates, [zeta_m, zeta_n, 3].
        :param zeta_b_dot: Bound grid velocities, [zeta_m, zeta_n, 3].
        :param zeta_w: Wake grid coordinates, [zeta_m_star, zeta_n, 3].
        :param gamma_b: Bound circulation strengths, [m, n].
        :param gamma_b_dot:  Bound circulation time derivatives, [m, n].
        :param gamma_w: Wake circulation strengths, [m_star, n].
        :param f_steady: Steady force contributions, [zeta_m, zeta_n, 3].
        :param f_unsteady: Unsteady force contributions, [zeta_m, zeta_n, 3].
        :param cs_ang: Control surface angle time histories, {name: []}.
        :param cs_vel: Control surface velocity time history, {name: []}.
        :param t: Time at the current step, [].
        :param i_ts: Time series index in full solution, [].
        :param dof_mapping: Array for mapping the aerodynamic grid onto the beam degrees of freedom, [zeta_n].
        :param static_horseshoe: If true, a horseshoe formulation was used to obtain the initial static solution.
        """

        self.zeta_b: Array = zeta_b
        self.zeta_b_dot: Array = zeta_b_dot
        self.zeta_w: Array = zeta_w
        self.gamma_b: Array = gamma_b
        self.gamma_b_dot: Array = gamma_b_dot
        self.gamma_w: Array = gamma_w
        self.f_steady: Array = f_steady
        self.f_unsteady: Array = f_unsteady
        self.cs_ang: dict[str, Array] = cs_ang
        self.cs_vel: dict[str, Array] = cs_vel
        self.surf_b_name: str = surf_b_name
        self.surf_w_name: str = surf_w_name
        self.i_ts: int = i_ts
        self.t: Array = t
        self.static_horseshoe: bool = static_horseshoe
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
