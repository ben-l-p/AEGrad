from __future__ import annotations

from typing import Sequence, Optional, Literal, Callable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from functools import partial

from jax import numpy as jnp
from jax import scipy as jsp
from jax import Array, vmap, jacrev

from aegrad.algebra.array_utils import block_axis
from aero.utils import (
    KernelFunction,
    mirror_grid,
    compute_c,
    compute_nc,
    compute_surf_nc,
    compute_surf_c,
    _steady_forcing,
)
from aegrad.print_utils import warn
from aegrad.algebra.base import finite_difference
from aegrad.algebra.array_utils import split_to_vertex
from aero.flowfields import FlowField
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
class DynamicAeroCase:
    r"""
    Class to hold time series of multiple aerodynamic surfaces.
    """

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: ArrayList,
        aic_lu: Optional[Array],
        aic_piv: Optional[Array],
        c: Optional[ArrayList],
        nc: Optional[ArrayList],
        gamma_b: ArrayList,
        gamma_b_dot: ArrayList,
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: ArrayList,
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
        self.c: ArrayList = c
        self.nc: ArrayList = nc
        self.gamma_b: ArrayList = gamma_b
        self.gamma_b_dot: ArrayList = gamma_b_dot
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: ArrayList = f_unsteady
        self.aic_lu: Array = aic_lu
        self.aic_piv: Array = aic_piv
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

        self.n_bound_panels: Sequence[int] = [
            g.size for g in self._gamma_b.index_all(0)
        ]
        self.n_bound_panels_tot: int = sum(self.n_bound_panels)

        self.n_wake_panels: Sequence[int] = [g.size for g in self._gamma_w.index_all(0)]
        self.n_wake_panels_tot: int = sum(self.n_wake_panels)

        self.n_bound_zeta_dofs: Sequence[int] = [
            g.size for g in self._zeta_b.index_all(0)
        ]
        self.n_bound_zeta_dofs_tot: int = sum(self.n_bound_zeta_dofs)

        self.n_wake_zeta_dofs: Sequence[int] = [
            g.size for g in self._zeta_w.index_all(0)
        ]
        self.n_wake_zeta_dofs_tot: int = sum(self.n_wake_zeta_dofs)

        self.gamma_b_slices: list[slice] = self._make_slices(self._gamma_b.index_all(0))
        self.zeta_dof_target_slices: list[slice] = self._make_slices(
            self._zeta_b.index_all(0)
        )
        self.zeta_dof_source_slices: list[slice] = self._make_slices(
            ArrayList([*self._zeta_b.index_all(0), *self._zeta_w.index_all(0)])
        )
        self.zeta_te_slices: list[Array] = self._make_te_slices(
            self._zeta_b.index_all(0)
        )
        self.gamma_te_slices: list[Array] = self._make_te_slices(
            self._gamma_b.index_all(0)
        )

        self._zeta_b_rings: Optional[ArrayList] = None
        self._zeta_w_rings: Optional[ArrayList] = None

    @property
    def zeta_b(self) -> ArrayList:
        return self._zeta_b

    @zeta_b.setter
    def zeta_b(self, zeta_b_list: ArrayList) -> None:
        self._zeta_b = zeta_b_list

    @property
    def zeta_b_rings(self) -> ArrayList:
        if self._zeta_b_rings is None:
            self._zeta_b_rings = ArrayList(
                [
                    vmap(self._grid_to_ring_decomp, 0, 0)(zeta_b)
                    for zeta_b in self._zeta_b
                ]
            )  # [n_surf][n_ts, m, n, 2, 2, 3]
        return self._zeta_b_rings

    @property
    def zeta_w_rings(self) -> ArrayList:
        if self._zeta_w_rings is None:
            self._zeta_w_rings = ArrayList(
                [
                    vmap(self._grid_to_ring_decomp, 0, 0)(zeta_w)
                    for zeta_w in self._zeta_w
                ]
            )  # [n_surf][n_ts, m_star, n, 2, 2, 3]
        return self._zeta_w_rings

    @property
    def n_zeta_dofs_tot(self) -> Sequence[int]:
        return [*self.n_bound_zeta_dofs, *self.n_wake_zeta_dofs]

    @property
    def zeta_b_dot(self) -> ArrayList:
        return self._zeta_b_dot

    @zeta_b_dot.setter
    def zeta_b_dot(self, zeta_b_dot_list: ArrayList) -> None:
        self._zeta_b_dot = zeta_b_dot_list

    @property
    def zeta_w(self) -> ArrayList:
        return self._zeta_w

    @zeta_w.setter
    def zeta_w(self, zeta_w_list: ArrayList) -> None:
        self._zeta_w = zeta_w_list

    @property
    def c(self) -> Optional[ArrayList]:
        return self._c

    @c.setter
    def c(self, c_list: Optional[ArrayList]) -> None:
        self._c = c_list

    @property
    def nc(self) -> Optional[ArrayList]:
        return self._nc

    @nc.setter
    def nc(self, nc_list: Optional[ArrayList]) -> None:
        self._nc = nc_list

    @property
    def gamma_b(self) -> ArrayList:
        return self._gamma_b

    @gamma_b.setter
    def gamma_b(self, gamma_b_list: ArrayList) -> None:
        self._gamma_b = gamma_b_list

    @property
    def gamma_b_dot(self) -> ArrayList:
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
        return self._f_unsteady

    @f_unsteady.setter
    def f_unsteady(self, f_unsteady_list: ArrayList) -> None:
        self._f_unsteady = f_unsteady_list

    @property
    def aic_lu(self) -> Optional[Array]:
        return self._aic_lu

    @aic_lu.setter
    def aic_lu(self, aic_lu_arr: Optional[Array]) -> None:
        self._aic_lu = aic_lu_arr

    @property
    def aic_piv(self) -> Optional[Array]:
        return self._aic_piv

    @aic_piv.setter
    def aic_piv(self, aic_piv_arr: Optional[Array]) -> None:
        self._aic_piv = aic_piv_arr

    @property
    def t(self) -> Array | float:
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

    def zeta_full(self, i_ts: int) -> ArrayList:
        return ArrayList(
            [*self.zeta_b.index_all(i_ts, ...), *self.zeta_w.index_all(i_ts, ...)]
        )

    def gamma_full(self, i_ts: int) -> ArrayList:
        return ArrayList(
            [*self.gamma_b.index_all(i_ts, ...), *self.gamma_w.index_all(i_ts, ...)]
        )

    def set_arraylist_at_ts(self, attr: str, values, i_ts: int) -> None:
        """Set each element of a stored ArrayList at timestep i_ts to values[i]."""
        arr = getattr(self, attr)
        for i_surf, val in enumerate(values):
            arr[i_surf] = arr[i_surf].at[i_ts, ...].set(val)

    def get_surf_snapshot(self, i_ts: int, i_surf: int) -> AeroSurfaceSnapshot:
        r"""
        Get snapshot for all aerodynamic surfaces.
        :param i_ts: Timestep index
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
            self._zeta_b_dot[i_surf] = (
                self._zeta_b_dot[i_surf]
                .at[i_ts, ...]
                .set(snapshot.zeta_b_dot[i_surf][0, ...])
            )
            self._zeta_w[i_surf] = (
                self._zeta_w[i_surf].at[i_ts, ...].set(snapshot.zeta_w[i_surf][0, ...])
            )
            self._gamma_b[i_surf] = (
                self._gamma_b[i_surf]
                .at[i_ts, ...]
                .set(snapshot.gamma_b[i_surf][0, ...])
            )
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
            self._f_unsteady[i_surf] = (
                self._f_unsteady[i_surf]
                .at[i_ts, ...]
                .set(snapshot.f_unsteady[i_surf][0, ...])
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
            for i_surf in range(self.n_surf):
                paths.extend(
                    self.get_surf_snapshot(i_ts=i_ts, i_surf=i_surf).plot(
                        directory, plot_bound=plot_bound, plot_wake=plot_wake
                    )
                )
        return paths

    def get_c(self, i_ts: int) -> ArrayList:
        r"""
        Get collocation points for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of collocation points, [n_surf][m, n, 3]
        """
        return self._c.index_all(i_ts, ...)

    def compute_c(self, i_ts: int) -> None:
        r"""
        Get normal vectors for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of normal vectors, [n_surf][m, n, 3]
        """
        # compute and set using helper
        c_list = compute_c(self._zeta_b.index_all(i_ts, ...))
        self.set_arraylist_at_ts("_c", c_list, i_ts)

    def compute_nc(self, i_ts: int) -> None:
        r"""
        Get normal vectors for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of normal vectors, [n_surf][m, n, 3]
        """
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

    def calculate_gamma_dot(self, i_ts: int, dt: Array) -> None:
        r"""
        Calculate time derivative of bound circulation strengths at specified time step using finite difference.
        :param i_ts: Timestep index
        :param dt: Time step size
        """

        def fd(arr):
            return finite_difference(i_ts, arr, dt, 0, order=1)

        for i_surf in range(self.n_surf):
            self._gamma_b_dot[i_surf] = (
                self._gamma_b_dot[i_surf].at[i_ts, ...].set(fd(self._gamma_b[i_surf]))
            )

    def calculate_unsteady_forcing(
        self,
        i_ts: int,
    ) -> None:
        r"""
        Calculate unsteady aerodynamic forcing for all surfaces at specified time step.
        :param i_ts: Timestep index
        """
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
            this_force = self._f_steady[i_surf][i_ts, ...]  # [ zeta_m, zeta_n, 3]
            if include_unsteady:
                this_force += self._f_unsteady[i_surf][i_ts, ...]

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
            )  # moment is r x_target f summed along strip [zeta_n, 3]
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
            rho * self._gamma_b_dot[i_surf][i_ts, ..., None] * nc, (0, 1)
        )

    def compute_aic_sys(
        self,
        x_source: ArrayList,
        x_target: ArrayList,
        nc: Optional[ArrayList],
        kernels: Sequence[KernelFunction],
    ) -> list[list[Array]]:
        """
        Compute the AIC matrix for a system of elements. Returns a list of AIC matrices, one for each element.
        :param x_source: List of source points to compute the AIC from, [n_surf][zeta_m, zeta_n, 3].
        :param x_target: List of target points to compute the AIC at, [n_surf][c_m, c_n, 3].
        :param nc: Bound normal vectors, [m, n, 3]. If None, no projection will be done.
        :param kernels: List of kernel functions to use for each source surface, [n_surf].
        :return: Nested sequences of AIC matrices, [n_target][n_source][c_m, c_n, c_m, c_n, 3], or [n_target][n_source][c_m, c_n, c_m, c_n] if projected onto normals.
        """

        aic_mats = []
        for i_target in range(len(x_target)):
            aic_mats.append([])
            for i_source in range(len(x_source)):
                # compute the AIC matrix, [n_cx, n_cy, n_ex, n_ey, 3]
                aic_ = self._compute_aic_grid(
                    c=x_target[i_target],
                    zeta=x_source[i_source],
                    kernel=kernels[i_source],
                )

                if self.mirror_point is not None and self.mirror_normal is not None:
                    # add influence from mirrored grid, if specified
                    zeta_mirror = mirror_grid(
                        zeta=x_source[i_source],
                        mirror_point=self.mirror_point,
                        mirror_normal=self.mirror_normal,
                    )
                    aic_ -= self._compute_aic_grid(
                        x_target[i_target], zeta_mirror, kernels[i_source]
                    )

                if nc is not None:
                    aic_ = jnp.einsum(
                        "ijklm,ijm->ijkl", aic_, nc[i_target]
                    )  # project onto normals
                aic_mats[-1].append(aic_)
        return aic_mats

    @staticmethod
    def add_wake_influence(
        aic_bs: list[list[Array]], aic_ws: list[list[Array]]
    ) -> list[list[Array]]:
        r"""
        Lump the wake influence onto the last column of the bound AIC matrices.
        :param aic_bs: Bound influence matrices, [][][c_m, c_n, zeta_m, zeta_n, 3].
        :param aic_ws: Wake influence matrices, [][][c_m, c_n, zeta_m_star, zeta_n, 3].
        :return: Updated bound influence matrices, [][][c_m, c_n, zeta_m, zeta_n, 3].
        """
        for i in range(len(aic_bs)):
            for j in range(len(aic_bs[i])):
                aic_bs[i][j] = (
                    aic_bs[i][j].at[:, :, -1, :].add(jnp.sum(aic_ws[i][j], axis=2))
                )
        return aic_bs

    @staticmethod
    def reshape_aic_sys(aic_mat: Array) -> Array:
        r"""
        Reshape an AIC matrix such that the source and target dimensions are flattened.
        :param aic_mat: Input AIC matrix, [c_m, c_n, zeta_m, zeta_n] or [c_m, c_n, zeta_m, zeta_n, 3].
        :return: Reshaped AIC matrix, [c_m*c_n, zeta_m*zeta_n] or [c_m*c_n, zeta_m*zeta_n, 3].
        """
        shape = aic_mat.shape
        new_shape = [shape[0] * shape[1], shape[2] * shape[3]]
        if len(shape) == 5:
            new_shape.append(shape[4])
        return aic_mat.reshape(new_shape)

    @classmethod
    def assemble_aic_sys(cls, aic_mats: Sequence[Sequence[Array]]) -> Array:
        r"""
        Assemble a nested sequence of AIC matrices into a single AIC matrix.
        :param aic_mats: Nested sequence of AIC matrices, [][][c_m, c_n, zeta_m, zeta_n] or [][][c_m, c_n, zeta_m, zeta_n, 3].
        :return: Assembled AIC matrix. [c_tot, zeta_tot] or [c_tot, zeta_tot, 3].
        """
        aic_mats_reshaped = [
            [cls.reshape_aic_sys(aic) for aic in aic_row] for aic_row in aic_mats
        ]
        return block_axis(aic_mats_reshaped, axes=(0, 1))

    def compute_aic_sys_assembled(
        self,
        i_ts: int,
        project_to_normals: bool,
        x_target: Optional[ArrayList] = None,
        custom_x_source: Optional[ArrayList] = None,
        custom_nc: Optional[ArrayList] = None,
        custom_kernels: Optional[Sequence[KernelFunction]] = None,
    ) -> Array:
        """
        Compute the assembled AIC matrix for a system of elements.
        :param i_ts: Timestep index to compute AIC for
        :param project_to_normals: If true, project the AIC onto the normal vectors at the target points.
        :param x_target: Optional custom target points to compute the AIC at, [n_surf][c_m, c_n, 3]. If None, will use
        collocation points.
        :param custom_x_source: Optional custom source points to compute the AIC from, [n_surf][zeta_m, zeta_n, 3]. If
        None, will use bound and wake grid points. If provided, custom kernels must also be passed.
        :param custom_nc: Optional custom normal vectors at source points, [n_surf][m, n, 3]. Only used if
        project_to_normals is True. If None, will use normals computed from bound grid points.
        :param custom_kernels: Optional custom kernels to use for each source surface, [n_surf]. Only used if
        custom_x_source is provided. If None, will use kernels stored in DynamicAeroCase.
        :return: Full AIC matrix, [c_tot, zeta_tot, 3], or [c_tot, zeta_tot] if projected onto normals.
        """
        aic_mats = self.compute_aic_sys(
            x_source=custom_x_source
            if custom_x_source is not None
            else self.zeta_full(i_ts=i_ts),
            x_target=x_target if x_target is not None else self._c.index_all(i_ts, ...),
            nc=None
            if not project_to_normals
            else (
                custom_nc if custom_nc is not None else self._nc.index_all(i_ts, ...)
            ),
            kernels=custom_kernels if custom_kernels is not None else self.kernels,
        )
        # reuse existing assembler which handles reshape internally
        return self.assemble_aic_sys(aic_mats)

    def get_v_ind(self, i_ts: int, x_target: ArrayList) -> ArrayList:
        aic_mat = self.compute_aic_sys_assembled(
            i_ts=i_ts, x_target=x_target, project_to_normals=False
        )
        v_vect = jnp.einsum(
            "ijk,j->ik", aic_mat, self.gamma_full(i_ts=i_ts).flatten()
        )  # [n_x_tot, 3]

        split_v = []
        cnt = 0
        for x_ in x_target:
            sz = x_.shape[0] * x_.shape[1]
            split_v.append(v_vect[cnt : cnt + sz, :].reshape(x_.shape))
            cnt += sz
        return ArrayList(split_v)

    def get_v_background(self, i_ts: int, x_target: ArrayList) -> ArrayList:
        r"""
        Get background velocity at specified points and time step
        :param t: Current time
        :param x_target: Points to evaluate background velocity at, [][..., 3]
        :return: Background velocity at points, [][..., 3]
        """
        return ArrayList(
            [
                self.flowfield.vmap_call(x_.reshape(-1, 3), self._t[i_ts]).reshape(
                    x_.shape
                )
                for x_ in x_target
            ]
        )

    def get_v_tot(self, i_ts: int, x: ArrayList) -> ArrayList:
        return self.get_v_ind(i_ts=i_ts, x_target=x) + self.get_v_background(
            i_ts=i_ts, x_target=x
        )

    @classmethod
    def _compute_aic_grid(
        cls,
        c: Array,
        zeta: Array,
        kernel: KernelFunction,
    ):
        """
        Compute the aerodynamic influence coefficient (AIC) across grids of points.
        :param c: Collocation points, [c_m, c_n, 3].
        :param zeta: Grid points, [zeta_m, zeta_n, 3].
        :param kernel: Kernel function to compute the influence.
        :return: AIC matrix, [c_m, c_n, zeta_m, zeta_n, 3].
        """

        # create the AIC in the spanwise and chordwise directions, and combine later
        # vectors in chordwise direction [zeta_m - 1, zeta_n, 2, 3]
        m_vect = jnp.stack((zeta[:-1, :, :], zeta[1:, :, :]), axis=-2)

        # vectors in spanwise direction [zeta_m, zeta_n - 1, 2, 3]
        n_vect = jnp.stack((zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2)

        # AIC matrices have one entry per filament
        m_aic = cls._aic_vmap(
            c, m_vect, kernel
        )  # chordwise AIC [m, n, zeta_m - 1, zeta_n, 3]
        n_aic = cls._aic_vmap(
            c, n_vect, kernel
        )  # spanwise AIC [m, n, zeta_m, zeta_n - 1, 3]

        return -jnp.diff(m_aic, axis=3) + jnp.diff(n_aic, axis=2)

    def aic_solve(self, i_ts: int, aic: Array, rhs: Array) -> Array:
        """
        Solve the AIC system for the circulation strengths.
        :param i_ts: Current time index
        :param aic: AIC matrix, [c_tot, zeta_tot].
        :param rhs: Right hand side vector, [c_tot].
        :return: Circulation strengths, [zeta_tot].
        """
        lu, piv = jsp.linalg.lu_factor(aic)
        self._aic_lu = self._aic_lu.at[i_ts, ...].set(lu)
        self._aic_piv = self._aic_piv.at[i_ts, ...].set(piv)
        return jsp.linalg.lu_solve((lu, piv), rhs)

    @staticmethod
    def _aic_vmap(
        c: Array,
        zeta: Array,
        kernel: KernelFunction,
    ) -> Array:
        """
        General AIC computation for any grid. Will vmap across first two dimensions.
        :param c: Collocation points, [c_m, c_n, 3].
        :param zeta: Grid points, [zeta_m, zeta_n, 3].
        :param kernel: Kernel function to compute the influence.
        :return: AIC matrix, [c_m, c_n, zeta_m, zeta_n, 3].
        """

        return vmap(
            vmap(vmap(vmap(kernel, (0, None), 0), (1, None), 1), (None, 0), 2),
            (None, 1),
            3,
        )(c, zeta)

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        return AeroSnapshot(
            zeta_b=self._zeta_b.index_all(i_ts, ...),
            zeta_b_dot=self._zeta_b_dot.index_all(i_ts, ...),
            zeta_w=self._zeta_w.index_all(i_ts, ...),
            c=self._c.index_all(i_ts, ...),
            nc=self._nc.index_all(i_ts, ...),
            gamma_b=self._gamma_b.index_all(i_ts, ...),
            gamma_b_dot=self._gamma_b_dot.index_all(i_ts, ...),
            gamma_w=self._gamma_w.index_all(i_ts, ...),
            f_steady=self._f_steady.index_all(i_ts, ...),
            f_unsteady=self._f_unsteady.index_all(i_ts, ...),
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            aic_lu=self._aic_lu[i_ts, ...],
            aic_piv=self._aic_piv[i_ts, ...],
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
    def from_static_case(cls, snapshot: AeroSnapshot, n_tstep: int) -> DynamicAeroCase:
        return snapshot.to_dynamic(i_ts=0, n_tstep=n_tstep)

    @property
    def source_rings(self) -> ArrayList:
        return ArrayList([*self.zeta_b_rings, *self.zeta_w_rings])

    @property
    def target_rings(self) -> ArrayList:
        return self.zeta_b_rings

    @property
    def source_zeta(self) -> ArrayList:
        return ArrayList([*self.zeta_b, *self.zeta_w])

    @property
    def target_zeta(self) -> ArrayList:
        return self.zeta_b

    def is_wake(self, i_surf: int) -> bool:
        return i_surf >= self.n_surf

    @staticmethod
    def _make_slices(arrs: ArrayList) -> list[slice]:
        slices = []
        cnt = 0
        for arr in arrs:
            sz = arr.size
            slices.append(slice(cnt, cnt + sz))
            cnt += sz
        return slices

    @staticmethod
    def _make_te_slices(arrs: ArrayList) -> list[Array]:
        slices = []
        cnt = 0
        for arr in arrs:
            sz = arr.size
            slices.append(jnp.arange(cnt, cnt + sz).reshape(arr.shape)[-1, ...].ravel())
            cnt += sz
        return slices

    @staticmethod
    def _grid_to_ring_decomp(
        zeta: Array,
    ) -> Array:
        r"""
        Decompose a grid into rings for AIC computation. This is needed for the Jacobian of the AIC with respect to the grid coordinates.
        :param zeta: [zeta_m, zeta_n, 3] grid coordinates
        :return: [m, n, 2, 2, 3] ring decomposition of the grid
        """
        return jnp.stack(
            (
                jnp.stack((zeta[:-1, :-1, :], zeta[1:, :-1, :]), axis=2),
                jnp.stack((zeta[:-1, 1:, :], zeta[1:, 1:, :]), axis=2),
            ),
            axis=3,
        )  # [m, n, 2, 2, 3]

    def _aic_entry(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute an AIC entry for a ring pair.
        :param source_ring: [2, 2, 3] coordinates of the source ring vertices
        :param target_ring: [2, 2, 3] coordinates of the target ring vertices
        :param kernel: kernel function
        :param project_to_normal: if true, the influence coefficient is projected onto the normal vector at the target ring
        :return: Influence coefficient for the target ring from the source ring projected onto the normal, [] or [3]
        depending on project_to_normal
        """
        c = compute_surf_c(target_ring)  # [3]
        aic = self._compute_aic_grid(c, source_ring, kernel)[0, 0, 0, 0, :]  # [3]
        if self.mirror_point is not None and self.mirror_normal is not None:
            source_ring_mirrored = mirror_grid(
                zeta=source_ring,
                mirror_point=self.mirror_point,
                mirror_normal=self.mirror_normal,
            )
            aic += self._compute_aic_grid(c, source_ring_mirrored, kernel)[
                0, 0, 0, 0, :
            ]  # [3]

        if project_to_normal:
            nc = compute_surf_nc(target_ring)[0, 0, :]  # [3]
            return jnp.inner(aic, nc)  # []
        else:
            return aic  # [3]

    def _d_aic_d_target(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute the Jacobian of the AIC entry with respect to the target ring coordinates.
        :param source_ring: [2, 2, 3]
        :param target_ring: [2, 2, 3]
        :param kernel: kernel function
        :return: [2, 2, 3] or [3, 2, 2, 3], derivatives with respect to target ring coordinates. Includes effect of
        perturbations in the normal vector if project_to_normal is true.
        """
        return jacrev(self._aic_entry, argnums=0)(
            target_ring,
            source_ring,
            kernel=kernel,
            project_to_normal=project_to_normal,
        )

    def _d_aic_d_source(
        self,
        target_ring: Array,
        source_ring: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> Array:
        r"""
        Compute the Jacobian of the AIC entry with respect to the source ring coordinates.
        :param source_ring: [2, 2, 3]
        :param target_ring: [2, 2, 3]
        :param kernel: kernel function
        :return: [2, 2, 3] or [3, 2, 2, 3], derivatives with respect to target ring coordinates. Includes effect of
        perturbations in the normal vector if project_to_normal is true.
        """
        return jacrev(self._aic_entry, argnums=1)(
            target_ring,
            source_ring,
            kernel=kernel,
            project_to_normal=project_to_normal,
        )

    @staticmethod
    def _ring_decomp_to_grid(
        aic_ring: Array,
        wrt: Literal["target", "source"],
    ) -> Array:
        r"""
        Recombine a ring-wise matrix back onto a full grid
        :param aic_ring: [m_t, n_t, m_s, n_s, 2, 2, 3] matrix defined on rings
        :return: [m_t, n_t, (m_t + 1), (n_t + 1), 3] panel scalars gradient with respect to target grid coordinates, or
        [m_t, n_t, (m_s + 1), (n_s + 1), 3] with respect to source grid coordinates.
        """
        m_t, n_t, m_s, n_s = aic_ring.shape[:4]
        match wrt:
            case "target":
                temp = aic_ring.sum(axis=(2, 3))  # [m_t, n_t, 2, 2, 3]
                aic_grid = jnp.zeros((m_t, n_t, m_t + 1, n_t + 1, 3))
                # Diagonal scatter: panel (i,j) only depends on its own ring vertices
                # ring vertex (a,b) of panel (i,j) maps to grid vertex (i+a, j+b)
                i_idx = jnp.arange(m_t)[:, None]  # [m_t, 1]
                j_idx = jnp.arange(n_t)[None, :]  # [1, n_t]
                aic_grid = aic_grid.at[i_idx, j_idx, i_idx, j_idx, :].add(
                    temp[:, :, 0, 0, :]
                )
                aic_grid = aic_grid.at[i_idx, j_idx, i_idx + 1, j_idx, :].add(
                    temp[:, :, 1, 0, :]
                )
                aic_grid = aic_grid.at[i_idx, j_idx, i_idx, j_idx + 1, :].add(
                    temp[:, :, 0, 1, :]
                )
                aic_grid = aic_grid.at[i_idx, j_idx, i_idx + 1, j_idx + 1, :].add(
                    temp[:, :, 1, 1, :]
                )

            case "source":
                aic_grid = jnp.zeros((m_t, n_t, m_s + 1, n_s + 1, 3))
                aic_grid = aic_grid.at[:, :, :-1, :-1, :].add(
                    aic_ring[:, :, :, :, 0, 0, :]
                )
                aic_grid = aic_grid.at[:, :, 1:, :-1, :].add(
                    aic_ring[:, :, :, :, 1, 0, :]
                )
                aic_grid = aic_grid.at[:, :, :-1, 1:, :].add(
                    aic_ring[:, :, :, :, 0, 1, :]
                )
                aic_grid = aic_grid.at[:, :, 1:, 1:, :].add(
                    aic_ring[:, :, :, :, 1, 1, :]
                )
            case _:
                raise ValueError(f"Invalid value for wrt: {wrt}")
        return aic_grid

    @classmethod
    def _d_aic_gamma(
        cls,
        gamma_b: Array,
        zeta_target_rings: Array,
        zeta_source_rings: Array,
        grad_func: Callable[[Array, Array], Array],
        wrt: Literal["target", "source"],
    ) -> Array:
        r"""
        Compute :math:`\frac{d \mathbf{A}}{d \zeta_b} \cdot \Gamma` for a surface pair bound grid. This is needed for the
        :param gamma_b: Bound circulation, [m, n]
        :param zeta_target_rings: Target rings, [m_t n_t, 2, 2, 3]
        :param zeta_source_rings: Source rings, [m_s, n_s, 2, 2, 3]
        :return: Derivative of AIC circulation product with respect to target and source bound grid coordinates,
        [m_t, n_t, zeta_target_m, zeta_target_n, 3] or [m_t, n_t, 3, zeta_target_m, zeta_target_n, 3], depending on normal
        projection.
        """

        d_aic = vmap(
            vmap(
                vmap(
                    vmap(
                        grad_func,
                        (0, None),
                        0,
                    ),
                    (1, None),
                    1,
                ),
                (None, 0),
                2,
            ),
            (None, 1),
            3,
        )(zeta_target_rings, zeta_source_rings)  # [m_t, n_t, m_s, n_s, 2, 2, 3]

        d_aic_gamma = jnp.einsum(
            "ijklmno,kl->ijklmno", d_aic, gamma_b
        )  # [m_t, n_t, m_s, n_s, 2, 2, 3]
        return cls._ring_decomp_to_grid(d_aic_gamma, wrt=wrt)

    def _d_aic_gamma_d_grids(
        self,
        gamma: Array,
        zeta_target_rings: Array,
        zeta_source_rings: Array,
        kernel: KernelFunction,
        project_to_normal: bool,
    ) -> tuple[Array, Array]:
        r"""
        Compute the contribution to the Jacobian of the AIC circulation product with respect to target and source bound grid
        coordinates.
        :param gamma: Circulation on the source grid, [m_s, n_s]
        :param zeta_target_rings: Target rings for the AIC computation, [m_t, n_t, 2, 2, 3]
        :param zeta_source_rings: Source rings for the AIC computation, [m_s, n_s, 2, 2, 3]
        :param kernel: Kernel function
        :return: AIC circulation product Jacobian contributions with respect to target and source bound grid coordinates,
        [m_t, n_t, zeta_target_m, zeta_target_n, 3] and [m_t, n_t, zeta_source_m, zeta_source_n, 3] if project to normals,
        otherwise [m_t, n_t, 3, zeta_target_m, zeta_target_n, 3] and [m_t, n_t, 3, zeta_source_m, zeta_source_n, 3]
        """

        d_aic_gamma_d_zeta_target = self._d_aic_gamma(
            zeta_target_rings=zeta_target_rings,
            zeta_source_rings=zeta_source_rings,
            gamma_b=gamma,
            grad_func=partial(
                self._d_aic_d_target,
                kernel=kernel,
                project_to_normal=project_to_normal,
            ),
            wrt="target",
        )
        d_aic_gamma_d_zeta_source = self._d_aic_gamma(
            zeta_target_rings=zeta_target_rings,
            zeta_source_rings=zeta_source_rings,
            gamma_b=gamma,
            grad_func=partial(
                self._d_aic_d_source,
                kernel=kernel,
                project_to_normal=project_to_normal,
            ),
            wrt="source",
        )
        return d_aic_gamma_d_zeta_target, d_aic_gamma_d_zeta_source

    def _d_v_bc_d_zeta(
        self,
        zeta_target_rings: Array,
    ) -> Array:
        r"""
        Compute the contribution to the Jacobian of the boundary condition velocity projected onto the normals, with respect
        to the target grid coordinates.
        :param zeta_target_rings: Target rings for the AIC computation, [m_t, n_t, 2, 2, 3]
        :return: Boundary condition velocity Jacobian contribution with respect to target bound grid coordinates,
        [m_t, n_t, 2, 2, 3]
        """

        def v_bc_entry(target_ring: Array) -> Array:
            c = compute_surf_c(target_ring)[0, 0, :]  # [3]
            nc = compute_surf_nc(target_ring)[0, 0, :]  # [3]
            v = self.flowfield(c, self.t)  # [3]
            return jnp.inner(v, nc)  # []

        def d_v_bc_entry_d_target(target_ring: Array) -> Array:
            return jacrev(v_bc_entry)(target_ring)  # [2, 2, 3]

        # we don't assemble this as it would be sparse
        return vmap(vmap(d_v_bc_entry_d_target, 0, 0), 1, 1)(
            zeta_target_rings
        )  # [m_t, n_t, 2, 2, 3]

    def d_gamma_b_d_zeta_b(
        self,
        static: bool,
        i_ts: int,
    ) -> Array:

        gamma = [*self.gamma_b, *self.gamma_w]

        if static:
            n_zeta_dofs = self.n_bound_zeta_dofs_tot
        else:
            raise NotImplementedError(
                "Dynamic case not implemented yet"
            )  # TODO: implement dynamic case

        d_gamma_b_zeta_b = jnp.zeros(
            (self.n_bound_panels_tot, n_zeta_dofs)
        )  # full matrix

        for i_target in range(self.n_surf):
            for i_source in range(2 * self.n_surf):
                d_aic_gamma_d_zeta_target, d_aic_gamma_d_zeta_source = (
                    self._d_aic_gamma_d_grids(
                        gamma=gamma[i_source],
                        zeta_target_rings=self.target_rings[i_target][i_ts, ...],
                        zeta_source_rings=self.source_rings[i_source][i_ts, ...],
                        kernel=self.kernels[i_source],
                        project_to_normal=True,
                    )
                )

                # shrink to wake if static
                if self.is_wake(i_source) and static:
                    d_aic_gamma_d_zeta_source = d_aic_gamma_d_zeta_source.sum(
                        axis=2, keepdims=True
                    )

                    this_source_zeta_dof_slice = self.zeta_te_slices[
                        i_source - self.n_surf
                    ]
                else:
                    this_source_zeta_dof_slice = self.zeta_dof_source_slices[i_source]

                # perturbations on the source rings
                d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                    self.gamma_b_slices[i_target], this_source_zeta_dof_slice
                ].add(
                    d_aic_gamma_d_zeta_source.reshape(self.n_bound_panels[i_target], -1)
                )

                # perturbations on the target rings
                d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                    self.gamma_b_slices[i_target], self.zeta_dof_target_slices[i_target]
                ].add(
                    d_aic_gamma_d_zeta_target.reshape(
                        self.n_bound_panels[i_target], self.n_bound_zeta_dofs[i_target]
                    )
                )

            # add pertubations in boundary condition (has no source)
            d_v_bc_d_zeta = self._d_v_bc_d_zeta(
                zeta_target_rings=self.zeta_b_rings[i_target][i_ts, ...],
            )  # [m_t, n_t, 2, 2, 3]

            m_t, n_t = self.gamma_b[i_target].shape
            assemble_d_v_bc = jnp.zeros((m_t, n_t, m_t + 1, n_t + 1, 3))

            # Diagonal scatter: panel (i,j) only depends on its own ring vertices
            i_idx = jnp.arange(m_t)[:, None]  # [m_t, 1]
            j_idx = jnp.arange(n_t)[None, :]  # [1, n_t]
            assemble_d_v_bc = assemble_d_v_bc.at[i_idx, j_idx, i_idx, j_idx, :].add(
                d_v_bc_d_zeta[:, :, 0, 0, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[i_idx, j_idx, i_idx + 1, j_idx, :].add(
                d_v_bc_d_zeta[:, :, 1, 0, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[i_idx, j_idx, i_idx, j_idx + 1, :].add(
                d_v_bc_d_zeta[:, :, 0, 1, :]
            )
            assemble_d_v_bc = assemble_d_v_bc.at[
                i_idx, j_idx, i_idx + 1, j_idx + 1, :
            ].add(d_v_bc_d_zeta[:, :, 1, 1, :])

            d_gamma_b_zeta_b = d_gamma_b_zeta_b.at[
                self.gamma_b_slices[i_target], self.zeta_dof_target_slices[i_target]
            ].add(assemble_d_v_bc.reshape(self.gamma_b[i_target].size, -1))

        return -jsp.linalg.lu_solve(
            (self._aic_lu[i_ts, ...], self._aic_piv[i_ts, ...]), d_gamma_b_zeta_b
        )

    def d_v_tot_d_gamma_b(self, i_ts: int, x: Array) -> Array:
        r"""
        Jacobian of total velocity at evaluation points wrt bound circulations.
        Since v_tot is linear in gamma, this is the bound-surface AIC columns.
        :param i_ts: Timestep index
        :param x: Evaluation points, [n_x, 3]
        :return: Jacobian, [n_x * 3, n_bound_panels_tot]
        """
        n_x = x.shape[0]
        zeta_full_list = ArrayList(
            [*self._zeta_b.index_all(i_ts, ...), *self._zeta_w.index_all(i_ts, ...)]
        )
        aic_mat = self.compute_aic_sys_assembled(
            i_ts=i_ts,
            x_target=ArrayList([x[:, None, :]]),
            custom_x_source=zeta_full_list,
            project_to_normals=False,
        )  # [n_x, n_gamma_full_tot, 3]
        aic_b = aic_mat[:, : self.n_bound_panels_tot, :]  # [n_x, n_bound_panels_tot, 3]
        # J[i*3+k, j] = aic_b[i, j, k]  =>  transpose axes (0,2,1) then reshape
        return aic_b.transpose(0, 2, 1).reshape(n_x * 3, self.n_bound_panels_tot)

    def d_v_tot_d_zeta(self, i_ts: int, x: ArrayList) -> Array:
        r"""
        Jacobian of total velocity at evaluation points wrt all grid coordinates.
        Background velocity does not depend on zeta, so only induced velocity contributes.
        :param i_ts: Timestep index
        :param x: Evaluation points, [n_surf][c_m, c_n, 3]
        :return: Jacobian, [n_x_tot * 3, n_zeta_dofs_tot]
        """
        gamma_flat = self.gamma_full(i_ts=i_ts).flatten()
        zeta_full = self.zeta_full(i_ts=i_ts)
        zeta_flat0 = zeta_full.flatten()
        zeta_shapes = zeta_full.shape

        def v_ind_from_zeta(zeta_flat: Array) -> Array:
            zeta_list = ArrayList.unravel(zeta_flat, zeta_shapes)
            aic_mat = self.compute_aic_sys_assembled(
                i_ts=i_ts,
                x_target=x,
                custom_x_source=zeta_list,
                project_to_normals=False,
            )  # [n_x_tot, n_gamma_full_tot, 3]
            return jnp.einsum("ijk,j->ik", aic_mat, gamma_flat).ravel()  # [n_x_tot * 3]

        return jacrev(v_ind_from_zeta)(zeta_flat0)  # [n_x_tot * 3, n_zeta_dofs_tot]

    def d_f_steady_d_gamma_b(self, i_ts: int, static: bool) -> Array:
        r"""
        Jacobian of steady forces wrt bound circulations.
        Forces are quadratic in gamma (gamma * v_tot(gamma)), so jacrev is used.
        For static VLM, wake perturbations are tied to trailing-edge perturbations,
        i.e. delta(gamma_w) = delta(gamma_te).
        :param i_ts: Timestep index
        :return: Jacobian, [n_f_tot, n_bound_panels_tot]
        """
        zeta_b = self._zeta_b.index_all(i_ts, ...)
        zeta_dot_b = self._zeta_b_dot.index_all(i_ts, ...)
        gamma_b = self._gamma_b.index_all(i_ts, ...)
        gamma_w = self._gamma_w.index_all(i_ts, ...)
        rho = self.flowfield.rho
        gamma_b_flat0 = gamma_b.flatten()
        gamma_b_shapes = gamma_b.shape
        # Pre-compute correctly-shaped source grids from _zeta_b/_zeta_w directly to
        # avoid double-indexing through the zeta_b property on AeroSnapshot.
        zeta_full_list = ArrayList([*zeta_b, *self._zeta_w.index_all(i_ts, ...)])

        def get_gamma_w_for_gamma_b(gamma_b_list: ArrayList) -> ArrayList:
            if not static:
                return gamma_w
            # Static VLM coupling: wake gamma perturbations follow TE bound perturbations.
            return ArrayList(
                [
                    jnp.broadcast_to(g_b[[-1], :], g_w.shape)
                    for g_b, g_w in zip(gamma_b_list, gamma_w)
                ]
            )

        def f_from_gamma_b(gamma_b_flat: Array) -> Array:
            gamma_b_list = ArrayList.unravel(gamma_b_flat, gamma_b_shapes)
            gamma_w_eff = get_gamma_w_for_gamma_b(gamma_b_list)
            gamma_full_flat = jnp.concatenate([gamma_b_flat, gamma_w_eff.flatten()])

            def v_func(x_: Array) -> Array:
                aic_mat = self.compute_aic_sys_assembled(
                    i_ts=i_ts,
                    x_target=ArrayList([x_]),
                    custom_x_source=zeta_full_list,
                    project_to_normals=False,
                )
                v_ind = jnp.einsum("ijk,j->ik", aic_mat, gamma_full_flat).reshape(
                    x_.shape
                )
                v_bg = self.flowfield.vmap_call(
                    x_.reshape(-1, 3), self._t[i_ts]
                ).reshape(x_.shape)
                return v_ind + v_bg

            f_list = _steady_forcing(
                zeta_b, zeta_dot_b, gamma_b_list, gamma_w_eff, v_func, None, rho
            )
            return jnp.concatenate([f.ravel() for f in f_list])

        return jacrev(f_from_gamma_b)(gamma_b_flat0)  # [n_f_tot, n_bound_panels_tot]

    def d_f_steady_d_zeta(self, i_ts: int, static: bool = False) -> Array:
        r"""
        Jacobian of steady forces wrt bound grid coordinates. This does not include effect of perturbing gamma.
        If static=True, wake grid perturbations are tied to trailing-edge perturbations.
        :param i_ts: Timestep index
        :param static: If True, include static wake-geometry coupling zeta_w(zeta_b)
        :return: Jacobian, [n_f_tot, n_bound_zeta_dofs_tot]
        """
        zeta_b = self._zeta_b.index_all(i_ts, ...)
        zeta_dot_b = self._zeta_b_dot.index_all(i_ts, ...)
        gamma_b = self._gamma_b.index_all(i_ts, ...)
        gamma_w = self._gamma_w.index_all(i_ts, ...)
        gamma_full_flat = jnp.concatenate([gamma_b.flatten(), gamma_w.flatten()])
        rho = self.flowfield.rho
        zeta_w = self._zeta_w.index_all(i_ts, ...)
        if static:
            # Keep each wake row offset relative to the trailing edge so zeta_w tracks zeta_te perturbations.
            wake_offsets = ArrayList(
                [
                    zeta_w[i_surf] - zeta_b[i_surf][[-1], :, :]
                    for i_surf in range(self.n_surf)
                ]
            )
        zeta_b_flat0 = zeta_b.flatten()
        zeta_b_shapes = zeta_b.shape

        def f_from_zeta_b(zeta_b_flat: Array) -> Array:
            zeta_b_list = ArrayList.unravel(zeta_b_flat, zeta_b_shapes)
            if static:
                zeta_w_eff = ArrayList(
                    [
                        zeta_b_list[i_surf][[-1], :, :] + wake_offsets[i_surf]
                        for i_surf in range(self.n_surf)
                    ]
                )
            else:
                zeta_w_eff = zeta_w

            zeta_full_list = ArrayList([*zeta_b_list, *zeta_w_eff])

            def v_func(x_: Array) -> Array:
                aic_mat = self.compute_aic_sys_assembled(
                    i_ts=i_ts,
                    x_target=ArrayList([x_]),
                    custom_x_source=zeta_full_list,
                    project_to_normals=False,
                )
                v_ind = jnp.einsum("ijk,j->ik", aic_mat, gamma_full_flat).reshape(
                    x_.shape
                )
                v_bg = self.flowfield.vmap_call(
                    x_.reshape(-1, 3), self._t[i_ts]
                ).reshape(x_.shape)
                return v_ind + v_bg

            f_list = _steady_forcing(
                zeta_b_list, zeta_dot_b, gamma_b, gamma_w, v_func, None, rho
            )
            return jnp.concatenate([f.ravel() for f in f_list])

        return jacrev(f_from_zeta_b)(zeta_b_flat0)  # [n_f_tot, n_bound_zeta_dofs_tot]

    def static_d_sol_d_zeta_b(
        self,
        i_ts: int,
    ) -> tuple[Array, Array]:
        r"""
        Total Jacobians of bound circulation and steady forces wrt bound grid coordinates.

        For gamma_b this is the direct sensitivity from thRe AIC system:
            d gamma_b / d zeta_b

        For f_steady the chain rule is applied to include the indirect path through gamma_b:
            d f_steady / d zeta_b |_total
                = d f_steady / d zeta_b |_partial
                + d f_steady / d gamma_b  @  d gamma_b / d zeta_b

        :param static: If True, treat the wake as collapsed to the trailing edge (static case).
        :param i_ts: Timestep index.
        :return: Tuple of
            d_gamma_b_d_zeta_b,  [n_bound_panels_tot, n_bound_zeta_dofs_tot]
            d_f_steady_d_zeta_b, [n_f_tot, n_bound_zeta_dofs_tot]
        """
        d_gamma_d_zeta = self.d_gamma_b_d_zeta_b(static=True, i_ts=i_ts)
        d_f_d_gamma = self.d_f_steady_d_gamma_b(static=True, i_ts=i_ts)
        d_f_d_zeta_partial = self.d_f_steady_d_zeta(i_ts=i_ts, static=True)
        d_f_d_zeta_total = d_f_d_zeta_partial + d_f_d_gamma @ d_gamma_d_zeta
        return d_gamma_d_zeta, d_f_d_zeta_total

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
            "n_bound_panels",
            "n_bound_panels_tot",
            "n_wake_panels",
            "n_wake_panels_tot",
            "n_bound_zeta_dofs",
            "n_bound_zeta_dofs_tot",
            "n_wake_zeta_dofs",
            "n_wake_zeta_dofs_tot",
            "gamma_b_slices",
            "zeta_dof_target_slices",
            "zeta_dof_source_slices",
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
            "_aic_lu",
            "_aic_piv",
            "_zeta_b_rings",
            "_zeta_w_rings",
            # these do not change after being initialised, however cannot have an Array marked as static
            "zeta_te_slices",
            "gamma_te_slices",
        )


@_make_pytree
class AeroSnapshot(DynamicAeroCase):
    r"""
    Class to hold snapshot of multiple aerodynamic surfaces at a single time step.

    This class subclasses DynamicAeroCase but internally stores all arrays with a
    leading time dimension of length 1 so that it can reuse all of
    DynamicAeroCase's methods. When users request a single-surface snapshot via
    indexing (snapshot[i_surf]) they receive an AeroSurfaceSnapshot with the
    time dimension removed for convenience.
    """

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: ArrayList,
        aic_lu: Optional[Array],
        aic_piv: Optional[Array],
        c: Optional[ArrayList],
        nc: Optional[ArrayList],
        gamma_b: ArrayList,
        gamma_b_dot: ArrayList,
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: ArrayList,
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
        Create an AeroSnapshot by wrapping per-snapshot arrays with a leading
        time dimension of size 1 so that DynamicAeroCase functions operate
        normally.
        """

        # call DynamicAeroCase initializer with expanded arrays
        super().__init__(
            zeta_b=zeta_b,
            zeta_b_dot=zeta_b_dot,
            zeta_w=zeta_w,
            aic_lu=aic_lu,
            aic_piv=aic_piv,
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
        return self._zeta_w.index_all(0, ...)

    @zeta_w.setter
    def zeta_w(self, value: ArrayList) -> None:
        self._zeta_w = value.index_all(None, ...)

    @property
    def c(self) -> Optional[ArrayList]:
        if self._c is None:
            return None
        else:
            return self._c.index_all(0, ...)

    @c.setter
    def c(self, value: Optional[ArrayList]) -> None:
        if value is None:
            self._c = None
        else:
            self._c = value.index_all(None, ...)

    @property
    def nc(self) -> Optional[ArrayList]:
        if self._nc is None:
            return None
        else:
            return self._nc.index_all(0, ...)

    @nc.setter
    def nc(self, value: Optional[ArrayList]) -> None:
        if value is None:
            self._nc = None
        else:
            self._nc = value.index_all(None, ...)

    @property
    def gamma_b(self) -> ArrayList:
        return self._gamma_b.index_all(0, ...)

    @gamma_b.setter
    def gamma_b(self, value: ArrayList) -> None:
        self._gamma_b = value.index_all(None, ...)

    @property
    def gamma_b_dot(self) -> ArrayList:
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
        return self._f_unsteady.index_all(0, ...)

    @f_unsteady.setter
    def f_unsteady(self, value: ArrayList) -> None:
        self._f_unsteady = value.index_all(None, ...)

    @property
    def aic_lu(self) -> Optional[Array]:
        if self._aic_lu is None:
            return None
        else:
            return self._aic_lu[0, ...]

    @aic_lu.setter
    def aic_lu(self, value: Optional[Array]) -> None:
        if value is None:
            self._aic_lu = None
        else:
            self._aic_lu = value[None, ...]

    @property
    def aic_piv(self) -> Optional[Array]:
        if self._aic_piv is None:
            return None
        else:
            return self._aic_piv[0, ...]

    @aic_piv.setter
    def aic_piv(self, value: Optional[Array]) -> None:
        if value is None:
            self._aic_piv = None
        else:
            self._aic_piv = value[None, ...]

    @property
    def t(self) -> Array | float:
        return self._t[0]

    @t.setter
    def t(self, t_val: Array | float) -> None:
        self._t = t_val

    @property
    def i_ts(self) -> Array | int:
        return self._i_ts

    @i_ts.setter
    def i_ts(self, i_ts_val: Array | int) -> None:
        self._i_ts = i_ts_val

    def to_dynamic(self, i_ts: int, n_tstep: int) -> DynamicAeroCase:
        """
        Expand this single-time snapshot into a DynamicAeroCase with n_tstep
        timesteps, placing the current snapshot at index i_ts (similar to the
        prior implementation).
        """

        def _expand_to_dyn(arr_list: ArrayList) -> ArrayList:
            out = []
            for a in arr_list:
                arr = jnp.zeros((n_tstep, *a.shape)).at[i_ts, ...].set(a)
                out.append(arr)
            return ArrayList(out)

        aic_lu_dyn = (
            jnp.zeros((n_tstep, *self.aic_lu.shape))
            .at[i_ts, ...]
            .set(self.aic_lu[0, ...])
        )
        aic_piv_dyn = (
            jnp.zeros((n_tstep, *self.aic_piv.shape))
            .at[i_ts, ...]
            .set(self.aic_piv[0, ...])
        )

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
            aic_lu=aic_lu_dyn,
            aic_piv=aic_piv_dyn,
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
        Return a single-surface snapshot with the time dimension removed for
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
        self, directory: str | PathLike, plot_bound: bool = True, plot_wake: bool = True
    ) -> Sequence[Path]:
        """
        Plot all aerodynamic surfaces in this single-time snapshot to VTU files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i_surf in range(self.n_surf):
            paths.extend(
                self[i_surf].plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)
            )
        return paths


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
