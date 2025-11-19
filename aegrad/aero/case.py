from __future__ import annotations

from functools import singledispatchmethod
from warnings import warn

from collections.abc import Sequence
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import fori_loop
from typing import Optional, Self
from os import PathLike
from pathlib import Path

from aegrad.utils import replace_self
from aegrad.aero.constants import HORSESHOE_LENGTH
from aegrad.array_utils import check_arr_dtype, neighbour_average, check_arr_shape
from aegrad.aero.uvlm_utils import GridDiscretization, AeroSnapshot
from aegrad.aero.flowfields import FlowField
from aegrad.aero.kernels import KernelFunction, biot_savart
from aegrad.aero.aic import compute_aic_sys_assembled, assemble_aic_sys, add_wake_influence, compute_aic_sys
from aegrad.algebra.base import finite_difference
from aegrad.algebra.se3 import vect_product as se3_vect_product


class AeroCase:
    def __init__(self,
                 n_tstep: int,
                 grid_shapes: Sequence[GridDiscretization | tuple[int, int, int]] | GridDiscretization | tuple[int, int, int],
                 variable_wake_disc: bool,
                 dof_mapping: Sequence[Array] | Array) -> None:
        r"""
        Initialize AeroCase class with all non-design parameters
        :param grid_shapes: Chordwise, spanwise, and wake panel discretizations for each aerodynamic surface, [n_surf][3]
        :param dof_mapping: Index mapping from aerodynamic surface nodes to global structure nodes, [n_surf][n]
        """

        # case for single inputs
        if isinstance(grid_shapes, (GridDiscretization, tuple[int, int, int])):
            grid_shapes = [grid_shapes]
        if isinstance(dof_mapping, Array):
            dof_mapping = [dof_mapping]

        # number of aerodynamic surfaces
        self.n_surf: int = len(grid_shapes)

        # set grid discretizations parameters for number of panels
        grid_disc = []

        for grid in grid_shapes:
            if isinstance(grid, Sequence):
                if len(grid) != 3:
                    raise ValueError("Grid shape tuple must have exactly three elements (m, n, m_star)")
                grid_disc.append(GridDiscretization(*grid))
            elif isinstance(grid, GridDiscretization):
                grid_disc.append(grid)
            else:
                raise TypeError("Grid shape must be either a Sequence of three integers or a GridDiscretization instance")
        self.grid_disc: tuple[GridDiscretization] = tuple(grid_disc)

        # count of number of panels
        self.n_bound_panels: tuple[int, ...] = tuple([gd.m * gd.n for gd in self.grid_disc])
        self.n_wake_panels: tuple[int, ...] = tuple([gd.m_star * gd.n for gd in self.grid_disc])
        self.n_panels_tot: int = sum(self.n_bound_panels) + sum(self.n_wake_panels)

        # placeholder for aerodynamic local grid coordinates, and global coordinates for wing and wake
        self._x0_b: Optional[list[Array]] = None
        self._zeta0_b: Optional[list[Array]] = None
        self._zeta0_w: Optional[list[Array]] = None

        self.gamma_b_slice, self.gamma_w_slice = self.make_gamma_slices()

        # store DOF mapping
        if len(dof_mapping) != self.n_surf:
            raise ValueError(f"Expected {self.n_surf} DOF mapping arrays, got {len(dof_mapping)}")
        for i_surf, map_ in enumerate(dof_mapping):
            check_arr_dtype(map_, int, "dof_mapping")
            check_arr_shape(map_, (self.grid_disc[i_surf].n + 1, ), "grid_disc")

        self.dof_mapping: tuple[Array, ...] = tuple(dof_mapping)

        # this must be optional as it is set as a design variable later
        self._flowfield: Optional[FlowField] = None

        # time step length
        self._dt: Optional[Array] = None

        # wake discretization parameters
        self.variable_wake_disc: bool = variable_wake_disc
        self._delta_w: Optional[Sequence[Array]] = None

        # kernel definitions per surface (seperate for wing and wake)
        self.kernels_b: Sequence[KernelFunction] = self.n_surf * [biot_savart]
        self.kernels_w: Sequence[KernelFunction] = self.n_surf * [biot_savart]

        # surface names used for plotting
        self.surf_b_names: list[str] = [f"surf_{i}_bound" for i in range(self.n_surf)]
        self.surf_w_names: list[str] = [f"surf_{i}_wake" for i in range(self.n_surf)]

        # placeholder for time domain case
        self.zeta_b: list[Array] = [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
        self.zeta_b_dot: list[Array] = [
            jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc
        ]
        self.zeta_w: list[Array] = [jnp.zeros((n_tstep, gd.m_star + 1, gd.n + 1, 3)) for gd in self.grid_disc]
        self.gamma_b: list[Array] = [jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc]
        self.gamma_b_dot: list[Array] = [jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc]
        self.gamma_w: list[Array] = [jnp.zeros((n_tstep, gd.m_star, gd.n)) for gd in self.grid_disc]
        self.f_steady: list[Array] = [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
        self.f_unsteady: list[Array] = [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
        self.n_tstep_tot: int = n_tstep
        self.t: Array = jnp.zeros(n_tstep)

    @property
    def flowfield(self) -> FlowField:
        if self._flowfield is None:
            raise ValueError("FlowField has not been set.")
        return self._flowfield

    @property
    def x0_b(self) -> list[Array]:
        if self._x0_b is None:
            raise ValueError("Design variable x0_b has not been set.")
        return self._x0_b

    @property
    def zeta0_b(self) -> list[Array]:
        if self._zeta0_b is None:
            raise ValueError("Design variable zeta0_b has not been set.")
        return self._zeta0_b

    @property
    def zeta0_w(self) -> list[Array]:
        if self._zeta0_w is None:
            raise ValueError("Design variable zeta0_w has not been set.")
        return self._zeta0_w

    @property
    def dt(self) -> Array:
        if self._dt is None:
            raise ValueError("Time step length dt has not been set.")
        return self._dt

    @property
    def delta_w(self) -> Sequence[Array]:
        if self._delta_w is None:
            raise ValueError("Wake displacement delta_w has not been set.")
        return self._delta_w

    def set_design_variables(self,
                             dt: float | Array,
                             flowfield: FlowField,
                             delta_w: Optional[Sequence[Array] | Array],
                             x0_aero: Sequence[Array] | Array,
                             hg0: Array) -> None:
        r"""
        Set aerodynamic design variables
        :param x0_aero: Aerodynamic local grid coordinates, [n_surf][m+1, n+1, 3]
        :param hg0: Beam global grid coordinates, [n, 4, 4]
        """
        if isinstance(delta_w, Array):
            delta_w = [delta_w]
        if isinstance(x0_aero, Array):
            x0_aero = [x0_aero]

        # set aerodynamic local coordinates
        if len(x0_aero) != self.n_surf:
            raise ValueError(f"Expected {self.n_surf} aerodynamic grid coordinate arrays, got {len(x0_aero)}")

        for i_surf in range(self.n_surf):
            check_arr_shape(x0_aero[i_surf], (self.grid_disc[i_surf].m + 1, self.grid_disc[i_surf].n + 1, 3), "x0_aero")
        self._x0_b = x0_aero

        # set global grid coordinates for bound and wake
        check_arr_shape(hg0, (None, 4, 4), "hg0")
        self._zeta0_b = self.hg_to_zeta(hg0)

        # set flowfield
        self._flowfield = flowfield

        # set timestep
        if isinstance(dt, float):
            self._dt = jnp.array(dt)
        elif isinstance(dt, Array):
            check_arr_shape(dt, (), "dt")
            self._dt = dt
        else:
            raise TypeError("dt must be either a float or an Array scalar")

        # set wake displacement
        self._delta_w = []
        if delta_w is not None:
            for i_surf, dw_ in enumerate(delta_w):
                check_arr_shape(dw_, (self.grid_disc[i_surf].m_star, 3), "delta_w")
                self._delta_w.append(jnp.concatenate((jnp.zeros((1, 3)), dw_), axis=0))
        else:
            # auto compute delta_w based on freestream and dt
            for i_surf in range(self.n_surf):
                self._delta_w.append(jnp.outer(jnp.arange(self.grid_disc[i_surf].m_star + 1),
                                               (self.flowfield.u_inf * self.dt)))
        self.initialise_wake()

    @replace_self
    def extend_n_tstep(self, n_tstep: int) -> Self:
        r"""
        Extend the time domain self arrays byto the specified number of time steps
        :param n_tstep: Number of time steps to extend to
        """
        warn("Extending number of timesteps may be slow")
        for var in (self.zeta_b, self.zeta_b_dot, self.zeta_w, self.gamma_b, self.gamma_w, self.f_steady, self.f_unsteady, self.t):
            for i_surf in range(self.n_surf):
                curr_val = var[i_surf]
                var[i_surf] = jnp.concatenate((curr_val, jnp.zeros((n_tstep, *curr_val.shape[1:]))))
        self.n_tstep_tot += n_tstep
        return self

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        r"""
        Get snapshot of aerodynamic surface at a single time step
        :param i_ts: Timestep index
        :return: AeroSnapshot at specified time step
        """

        if i_ts < 0 or i_ts >= self.n_tstep_tot:
            raise IndexError("Timestep index out of range")

        return AeroSnapshot(
            zeta_b=[self.zeta_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            zeta_b_dot=[self.zeta_b_dot[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            zeta_w=[self.zeta_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            gamma_b=[self.gamma_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            gamma_w=[self.gamma_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            f_steady=[self.f_steady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            f_unsteady=[self.f_unsteady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)],
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=float(self.t[i_ts]),
            n_surf=self.n_surf
        )

    def get_v_ind(self, x: Array, i_ts: int) -> Array:
        """
        Compute the induced velocity at points x due to a system of vortex elements.
        :param x: Points at which to compute induced velocity, [..., 3]
        :param i_ts: Timestep index
        :return: Induced velocity at points x, of shape [..., 3].
        """

        x_flat = x.reshape(-1, 3)  # [n_x, 3]
        gamma_flat = self.get_gamma_vect(i_ts)

        aic = compute_aic_sys_assembled(
            [x_flat],
            self.get_zeta(i_ts),
            [*self.kernels_b, *self.kernels_w],
        )  # shape [n_x, n_gamma, 3]

        return jnp.einsum("ijk,j->ik", aic, gamma_flat).reshape(x.shape)

    def get_v_background(self, i_ts: int, x: Array) -> Array:
        r"""
        Get background (freestream) velocity at specified points and time step
        :param i_ts: Timestep index
        :param x: Points to evaluate background velocity at, [n_points, 3]
        :return: Background velocity at points, [n_points, 3]
        """
        return self.flowfield.vmap_call(x, self.t[i_ts])


    def get_v_tot(self, i_ts: int, x: Array) -> Array:
        r"""
        Get induced velocity at specified points and time step by the elements in the flow and the freesteam
        :param i_ts: Timestep index
        :param x: Points to evaluate induced velocity at, [n_points, 3]
        :return: Induced velocity at points, [n_points, 3]
        """
        return self.get_v_ind(x, i_ts) + self.get_v_background(i_ts, x)


    def get_zeta_te_surf(self, i_ts: int, i_surf: int) -> Array:
        r"""
        Get trailing edge grid coordinates for a single surface at specified time step
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :return: Trailing edge grid coordinates, [zeta_n, 3]
        """
        return self.zeta_b[i_surf][i_ts, -1, ...]


    def get_gamma_te_surf(self, i_ts: int, i_surf: int) -> Array:
        r"""
        Get trailing edge circulation strengths for a single surface at specified time step
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :return: Trailing edge circulation strengths, [gamma_n]
        """
        return self.gamma_b[i_surf][i_ts, -1, :]

    def get_zeta_b(self, i_ts: int) -> list[Array]:
        r"""
        Get bound grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        """
        return [self.zeta_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]

    def get_zeta_w(self, i_ts: int) -> list[Array]:
        r"""
        Get wake grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        """
        return [self.zeta_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]

    def get_zeta(self, i_ts: int) -> list[Array]:
        r"""
        Get full grid coordinates (bound, wake) for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of grid coordinates for each surface, [2 * n_surf][m_, n_, 3]
        """
        return self.get_zeta_b(i_ts) + self.get_zeta_w(i_ts)

    def hg_to_zeta(self, hg: Array) -> Sequence[Array]:
        zetas = []
        for i_surf in range(self.n_surf):
            this_hg = jnp.take(hg, self.dof_mapping[i_surf], axis=0)   # [n, 4, 4]
            # zetas.append(vmap(vmap(se3_vect_product, (0, 1), 1), (None, 0), 0)(this_hg, self.x0_b[i_surf]))
            zetas.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg, self.x0_b[i_surf]
                )
            )
        return zetas

    def hg_dot_to_zeta_dot(self, hg_dot: Array) -> Sequence[Array]:
        zeta_dots = []
        for i_surf in range(self.n_surf):
            this_hg_dot = jnp.take(hg_dot, self.dof_mapping[i_surf], axis=0)   # [n, 4, 4]
            zeta_dots.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg_dot, self.x0_b[i_surf]
                )
            )
        return zeta_dots

    def get_gamma_b_vect(self, i_ts: int) -> Array:
        r"""
        Get bound circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Bound circulation strengths vector, [gamma_b_tot]
        """
        return jnp.concatenate([gamma_[i_ts, ...].ravel() for gamma_ in self.gamma_b], axis=0)

    def get_gamma_w_vect(self, i_ts: int) -> Array:
        r"""
        Get wake circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Wake circulation strengths vector, [gamma_w_tot]
        """
        return jnp.concatenate([gamma_[i_ts, ...].ravel() for gamma_ in self.gamma_w], axis=0)

    def get_gamma_vect(self, i_ts: int) -> Array:
        r"""
        Get total circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Total circulation strengths vector, [gamma_tot]
        """
        return jnp.concatenate([self.get_gamma_b_vect(i_ts), self.get_gamma_w_vect(i_ts)], axis=0)

    @replace_self
    def set_gamma_b(self, gamma_vec: Array, i_ts: int) -> Self:
        r"""
        Set bound circulation strengths from total circulation strengths vector at specified time step
        :param gamma_vec: Total circulation strengths vector, [gamma_b_tot]
        :param i_ts: Timestep index
        :return: List of wake circulation strengths for each surface, [n_surf][m_star, n]
        """
        for i_surf in range(self.n_surf):
            self.gamma_b[i_surf] = self.gamma_b[i_surf].at[i_ts, ...].set(gamma_vec[self.gamma_b_slice[i_surf]].reshape(self.grid_disc[i_surf].m, self.grid_disc[i_surf].n))
        return self

    @replace_self
    def set_zeta_b(self, zeta_b: Sequence[Array], i_ts: int) -> Self:
        r"""
        Set bound grid coordinates from list of grid coordinates at specified time step
        :param zeta_b: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b[i_surf] = self.zeta_b[i_surf].at[i_ts, ...].set(zeta_b[i_surf])
        return self

    @replace_self
    def set_zeta_b_dot(self, zeta_b_dot: Sequence[Array], i_ts: int) -> Self:
        r"""
        Set bound grid velocities from list of grid coordinates at specified time step
        :param zeta_b_dot: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b_dot[i_surf] = self.zeta_b_dot[i_surf].at[i_ts, ...].set(zeta_b_dot[i_surf])
        return self

    @replace_self
    def set_zeta_w(self, zeta_w: Sequence[Array], i_ts: int) -> Self:
        r"""
        Set bound grid coordinates from list of grid coordinates at specified time step
        :param zeta_w: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_w[i_surf] = self.zeta_w[i_surf].at[i_ts, ...].set(zeta_w[i_surf])
        return self

    @singledispatchmethod
    @replace_self
    def set_gamma_w(self, gamma_vec: Array, i_ts: int) -> Self:
        r"""
        Set wake circulation strengths from total circulation strengths vector at specified time step
        :param gamma_vec: Total circulation strengths vector, [gamma_w_tot]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.gamma_w[i_surf] = self.gamma_w[i_surf].at[i_ts, ...].set(
                gamma_vec[self.gamma_w_slice[i_surf]].reshape(self.grid_disc[i_surf].m_star, self.grid_disc[i_surf].n))
        return self

    @set_gamma_w.register(Sequence)
    @replace_self
    def _(self, gamma_list: Sequence[Array], i_ts: int) -> Self:
        for i_surf in range(self.n_surf):
            self.gamma_w[i_surf] = self.gamma_w[i_surf].at[i_ts, ...].set(gamma_list[i_surf])
        return self

    @replace_self
    def set_gamma_w_static(self, i_ts: int) -> Self:
        r"""
        Set wake circulation strengths from horseshoe wake vector
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.gamma_w[i_surf] = self.gamma_w[i_surf].at[i_ts, ...].set(self.get_gamma_te_surf(i_ts, i_surf)[None, :])
        return self

    def make_gamma_slices(self) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        r"""
        Create slices for the bound and wake circulation strengths in their respective solution vectors based on the grid
        discretizations.
        :return: Slices for bound and wake circulation strengths.
        """
        gamma_b_slices = []
        gamma_w_slices = []
        cnt_b = 0
        cnt_w = 0
        for grid_disc in self.grid_disc:
            n_b = grid_disc.m * grid_disc.n
            n_w = grid_disc.m_star * grid_disc.n
            gamma_b_slices.append(slice(cnt_b, cnt_b + n_b))
            cnt_b += n_b
            gamma_w_slices.append(slice(cnt_w, cnt_w + n_w))
            cnt_w += n_w
        return tuple(gamma_b_slices), tuple(gamma_w_slices)

    def make_surf_horseshoe_wake(
            self, i_ts: int, i_surf: int, horseshoe_length: float
    ) -> Array:
        r"""
        Create a horseshoe wake for a given surface at a given time step.
        :param self:
        :param i_ts:
        :param i_surf:
        :param horseshoe_length:
        :return:
        """
        zeta_te = self.get_zeta_te_surf(i_ts, i_surf)   # [zeta_n, 3]
        wake_end = zeta_te + (self.flowfield.u_inf_dir * horseshoe_length)[None, :] # [zeta_n, 3]
        return jnp.stack((zeta_te, wake_end), axis=0)

    @staticmethod
    def get_surf_c_nc(zeta: Array) -> tuple[Array, Array]:
        r"""
        Compute the colocation points and normals for a given grid of points.
        :param zeta: Grid of points, [zeta_m, zeta_n, 3]
        :return: Colocation points and normals, [zeta_m-1, zeta_n-1, 3], [zeta_m-1, zeta_n-1, 3]
        """
        c = neighbour_average(zeta, axes=(0, 1))

        # find area normal as the cross-product of the diagonals for each rectilinear panel
        diag1 = zeta[1:, 1:, :] - zeta[:-1, :-1, :]  # [n_sx, n_cy, 3]
        diag2 = zeta[1:, :-1, :] - zeta[:-1, 1:, :]
        nc = jnp.cross(diag1, diag2)
        return c, nc

    @staticmethod
    def get_c_nc(zetas: Sequence[Array]) -> tuple[Sequence[Array], Sequence[Array]]:
        r"""
        Compute the colocation points and normals for a given list of grids of points.
        :param zetas: List of grids of points, [n_surf][zeta_m, zeta_n, 3]
        :return: List of colocation points and normals for each surface, [n_surf][zeta_m-1, zeta_n-1, 3], [n_surf][zeta_m-1, zeta_n-1, 3]
        """
        cs, ncs = [], []
        for zeta in zetas:
            c, nc = AeroCase.get_surf_c_nc(zeta)
            cs.append(c)
            ncs.append(nc)
        return cs, ncs

    @replace_self
    def initialise_wake(self, zeta_te_input: Optional[Sequence[Array]] = None) -> Self:
        r"""
        Generate initial wake grid coordinates
        :param zeta_te_input: Optional trailing edge coordinates to use instead of bound grid trailing edge, [n_surf][n+1, 3]
        Note that the wake grid does not include the trailing edge, assuming that delta_w[0, :] != 0
        """

        self._zeta0_w = []
        for i_surf, this_delta_w in enumerate(self.delta_w):
            # get bound grid coordinates
            if zeta_te_input is not None:   # optional input for trailing edge, should ever be required
                zeta_te = zeta_te_input[i_surf]   # [n+1, 3]
                check_arr_shape(zeta_te, (self.grid_disc[i_surf].n + 1, 3), None)
            else:
                zeta_te = self.zeta0_b[i_surf][-1, :, :]   # [n+1, 3]

            # set wake grid coordinates as trailing edge + displacement
            self._zeta0_w.append(zeta_te[None, ...] + this_delta_w[:, None, :])
        return self

    def propagate_surf_wake(self, i_ts: int, i_surf: int, free_wake: bool) -> tuple[Array, Array]:
        r"""
        Convect the wake at some given velocity. This step includes convection from the trailing edge and culling the
        downstream data.
        :param i_ts: Time step index
        :param free_wake: Whether the wake is free or not. If False, the induced velocity is not added.
        :param i_surf: Surface index
        :return: New wake grid and circulation, [zeta_w_m, zeta_n, 3], [zeta_w_m-1, zeta_n]
        """
        zeta_te = self.get_zeta_te_surf(i_ts, i_surf)   # [zeta_n, 3]
        gamma_te = self.get_gamma_te_surf(i_ts, i_surf)  # [zeta_n]

        # variable wake discretisation also depends on the final element
        if self.variable_wake_disc:
            zeta_base = self.zeta_w[i_surf][i_ts - 1, ...]  # [zeta_w_m, zeta_n, 3]
            gamma_base = self.gamma_w[i_surf][i_ts - 1, ...] # [gamma_w_m, gamma_n]
        else:
            zeta_base = self.zeta_w[i_surf][i_ts - 1, :-1, ...]  # [zeta_w_m-1, zeta_n, 3]
            gamma_base = self.gamma_w[i_surf][i_ts - 1, :-1, ...]  # [gamma_w_m-1, gamma_n]

        # values we wish to propagate
        zeta_pre = jnp.concatenate(
            (zeta_te[None, ...], zeta_base), axis=0
        ) # [zeta_w_m+1 | zeta_w_m, zeta_n, 3]
        gamma_pre = jnp.concatenate(
            (gamma_te[None, ...], gamma_base), axis=0
        )   # [gamma_w_m+1 | gamma_w_m, gamma_n]

        # add the element induced velocity if the wake is free
        if free_wake:
            v = self.get_v_tot(i_ts - 1, zeta_pre)
        else:
            v = self.get_v_background(i_ts - 1, zeta_pre)

        # find the integrated in time version
        zeta_w_new = zeta_pre + self.dt * v
        gamma_w_new = gamma_pre

        if self.variable_wake_disc:
            zeta_pre_redisc = jnp.concatenate((zeta_te[None, :], zeta_w_new), axis=0)  # [zeta_w_m+2, zeta_n, 3]
            gamma_pre_redisc = jnp.concatenate((gamma_te[None, :], gamma_base), axis=0)  # [gamma_w_m+2, gamma_n]

            # if the wake discretisation is variable, we need to rediscretize the wake
            s_zeta = jnp.concatenate(
                (
                    jnp.zeros((1, zeta_te.shape[0])),  # [1, zeta_n]
                    jnp.cumsum(
                        jnp.linalg.norm(zeta_pre_redisc[1:, ...] - zeta_pre_redisc[:-1, ...], axis=-1), # [zeta_w_m+1, zeta_n]
                        axis=0,
                    ),  # [zeta_w_m, zeta_n]
                ),
                axis=0,
            )   # distance along each wake filament for each point [zeta_w_m + 1, zeta_n]

            # consider gamma to be at midpoints of zeta
            s_gamma = neighbour_average(s_zeta, axes=(0, 1)) # [gamma_w_m + 1, gamma_w_n]

            # coordinates along desired discretized streamline, [zeta_w_m]
            s_base = jnp.cumsum(jnp.linalg.norm(self.delta_w[i_surf], axis=-1), axis=0)

            zeta_w_new = vmap(
                vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=0),
                in_axes=(None, None, 2),
                out_axes=2,
            )(s_base, s_zeta, zeta_pre_redisc)

            gamma_w_new = vmap(
                vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=0),
                in_axes=(None, None, 2),
                out_axes=2,
            )(s_base, s_gamma, gamma_pre_redisc)

        return zeta_w_new, gamma_w_new

    @replace_self
    def propagate_wake(self, i_ts: int, free_wake: bool) -> Self:
        r"""
        Convect the wake at some given velocity for all surfaces.
        :param i_ts: Time step index
        :param free_wake: Whether the wake is free or not. If False, the induced velocity is not added.
        :return: New wake grid and circulation, [zeta_w_m, zeta_n, 3], [zeta_w_m-1, zeta_n]
        """

        for i_surf in range(self.n_surf):
            zeta_w_surf, gamma_w_surf = self.propagate_surf_wake(
                i_ts,
                i_surf,
                free_wake,
            )
            self.zeta_w[i_surf] = self.zeta_w[i_surf].at[i_ts, ...].set(zeta_w_surf)
            self.gamma_w[i_surf] = self.gamma_w[i_surf].at[i_ts, ...].set(gamma_w_surf)
        return self

    @replace_self
    def calculate_gamma_dot(self, i_ts: int) -> Self:
        for i_surf in range(self.n_surf):
            self.gamma_b_dot[i_surf] = self.gamma_b_dot[i_surf].at[i_ts, ...].set(
                finite_difference(i_ts, self.gamma_b[i_surf], self.dt, 0, order=1)
            )
        return self

    @replace_self
    def calculate_steady_forcing(self, i_ts: int) -> Self:
        return self

    @replace_self
    def calculate_unsteady_forcing(self, i_ts: int) -> Self:
        return self

    @replace_self
    def solve(
        self,
        i_ts: int,
        hg: Optional[Array],
        hg_dot: Optional[Array],
        static: bool,
        free_wake: bool,
        horseshoe: bool,
    ) -> Self:
        if not static and horseshoe:
            warn(
                "Horseshoe wake not compatible with non-static solve. Overriding horseshoe to False."
            )
            horseshoe = False

        zetas_b = self.zeta0_b if hg is None else self.hg_to_zeta(hg)
        self.set_zeta_b(zetas_b, i_ts)

        cs, ncs = self.get_c_nc(zetas_b)

        if hg_dot is None:
            c_dot = [jnp.zeros_like(c) for c in cs]
        else:
            zeta_b_dot = self.hg_dot_to_zeta_dot(hg_dot)

            self.set_zeta_b_dot(zeta_b_dot, i_ts)


            c_dot = [
                neighbour_average(zeta_dot, axes=(0, 1))
                for zeta_dot in zeta_b_dot
            ]

        if static:
            # initialise wake
            if horseshoe:
                zeta_ws = [
                    self.make_surf_horseshoe_wake(i_ts, i_surf, HORSESHOE_LENGTH)
                    for i_surf in range(self.n_surf)
                ]
            else:
                # use initialised
                zeta_ws = self.zeta0_w
        else:
            # propagate wake
            self.propagate_wake(i_ts, free_wake)
            zeta_ws = self.get_zeta_w(i_ts)

        if not horseshoe:
            self.set_zeta_w(zeta_ws, i_ts)
        else:
            # set wake to look like a shortened version of the full horseshoe
            zeta_w_stretched = self.zeta0_w
            for i_surf in range(self.n_surf):
                zeta_w_stretched[i_surf] = zeta_w_stretched[i_surf].at[:-1, ...].set(zeta_w_stretched[i_surf][0, ...])
            self.set_zeta_w(zeta_w_stretched, i_ts)

        # compute AIC matrix for bound-bound interactions, [c_tot, gamma_b_tot]
        aic_blocks = compute_aic_sys(cs, zetas_b, self.kernels_b, ncs)

        # compute AIC matrix for bound-wake interactions, [c_tot, gamma_w_tot]
        aic_w_blocks = compute_aic_sys(cs, zeta_ws, self.kernels_w, ncs)

        if static:
            # lump wake influence onto trailing edge strengths
            aic_blocks = add_wake_influence(aic_blocks, aic_w_blocks)

        aic = assemble_aic_sys(aic_blocks)

        cs_vector = jnp.concatenate([c.reshape(-1, 3) for c in cs], axis=0)
        c_dot_vector = jnp.concatenate([cd.reshape(-1, 3) for cd in c_dot], axis=0)
        v_bc = c_dot_vector - self.get_v_background(i_ts, cs_vector)

        if not static:
            aic_w = assemble_aic_sys(aic_w_blocks)
            v_bc -= aic_w @ self.get_gamma_w_vect(i_ts)
        n_vect = jnp.concatenate([nc.reshape(-1, 3) for nc in ncs], axis=0)
        v_bc_n = jnp.einsum('ij,ij->i', v_bc, n_vect)

        gamma_b_vect: Array = jnp.linalg.solve(aic, v_bc_n)
        self.set_gamma_b(gamma_b_vect, i_ts)

        if static:
            self.set_gamma_w_static(i_ts)
        else:
            self.calculate_gamma_dot(i_ts)
            self.calculate_unsteady_forcing(i_ts)
        self.calculate_steady_forcing(i_ts)

        return self

    @replace_self
    def solve_static(self, i_ts: int, hg: Optional[Array], horseshoe: bool) -> Self:
        self.solve(i_ts, hg, None, static=True, free_wake=False, horseshoe=horseshoe)
        return self

    @replace_self
    def solve_prescribed_dynamic(self, i_ts_start: int, hg: Array, hg_dot: Array, free_wake: bool) -> Self:
        check_arr_shape(hg, (None, None, 4, 4), "hg")
        if hg_dot is not None:
            if hg.shape != hg_dot.shape:
                raise ValueError(
                    f"hg_dot must have the same shape as hg, got {hg_dot.shape} vs {hg.shape}"
                )
        n_tstep = hg.shape[0]
        return fori_loop(
            i_ts_start,
            i_ts_start + n_tstep,
            lambda i_ts, case: AeroCase.solve(
                case,
                i_ts,
                hg,
                hg_dot,
                static=False,
                free_wake=free_wake,
                horseshoe=False,
            ),
            init_val=self,
        )

    def plot(self, directory: PathLike, index: Optional[slice | Sequence[int] | Array] = None, plot_wake: bool = True) -> None:
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep_tot)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif index is None:
            index_ = jnp.arange(self.n_tstep_tot)
        else:
            raise TypeError("index must be a slice, sequence of ints, or Array")

        for i_ts in index_:
            snapshot = self[i_ts]
            # TODO: add PVD writer
            paths = snapshot.plot(directory, plot_wake=plot_wake)

    def reference_snapshot(self) -> AeroSnapshot:
        return AeroSnapshot(
            zeta_b=self.zeta0_b,
            zeta_b_dot=[jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc],
            zeta_w=self.zeta0_w,
            gamma_b=[jnp.zeros((gd.m, gd.n)) for gd in self.grid_disc],
            gamma_w=[jnp.zeros((gd.m_star, gd.n)) for gd in self.grid_disc],
            f_steady=[jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc],
            f_unsteady=[jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc],
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=-1,
            t=0.0,
            n_surf=self.n_surf
        )

    def plot_reference(self, directory: PathLike, plot_wake: bool = True) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(directory, plot_wake=plot_wake)


