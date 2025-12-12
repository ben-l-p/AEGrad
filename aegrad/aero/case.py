from __future__ import annotations
from functools import singledispatchmethod
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import fori_loop
from typing import Optional, Self
from os import PathLike
from pathlib import Path

from aegrad.utils import replace_self, make_pytree
from aegrad.algebra.test_routines import check_if_all_se3_g, check_if_all_se3_a
from aegrad.aero.uvlm_utils import get_c, get_nc, propagate_wake, steady_forcing
from aegrad.aero.constants import HORSESHOE_LENGTH
from aegrad.algebra.array_utils import check_arr_dtype, neighbour_average, check_arr_shape, split_to_vertex, ArrayList
from aegrad.aero.data_structures import GridDiscretization, AeroSnapshot
from aegrad.aero.flowfields import FlowField
from aegrad.aero.kernels import KernelFunction, biot_savart_epsilon, biot_savart, biot_savart_cutoff
from aegrad.aero.aic import compute_aic_sys_assembled, assemble_aic_sys, add_wake_influence, compute_aic_sys
from aegrad.algebra.base import finite_difference
from aegrad.algebra.se3 import vect_product as se3_vect_product
from aegrad.aero.linear import LinearAero, LinearWakeType
from aegrad.print_output import print_with_time, warn, jax_print
from aegrad.plotting.pvd import write_pvd

@make_pytree
class AeroCase:
    def __init__(self,
                 n_tstep: int,
                 grid_shapes: Sequence[GridDiscretization | tuple[int, int, int]] | GridDiscretization | tuple[int, int, int],
                 variable_wake_disc: bool,
                 dof_mapping: ArrayList | Sequence[Array] | Array,
                 kernel: Optional[KernelFunction] = None) -> None:
        r"""
        Initialize AeroCase class with all non-design parameters
        Initialize AeroCase class with all non-design parametersA
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
                    raise ValueError("Grid shapes tuple must have exactly three elements (m, n, m_star)")
                grid_disc.append(GridDiscretization(*grid))
            elif isinstance(grid, GridDiscretization):
                grid_disc.append(grid)
            else:
                raise TypeError("Grid shapes must be either a Sequence of three integers or a GridDiscretization instance")
        self.grid_disc: tuple[GridDiscretization] = tuple(grid_disc)

        # count of number of panels
        self.n_bound_panels: tuple[int, ...] = tuple([gd.m * gd.n for gd in self.grid_disc])
        self.n_wake_panels: tuple[int, ...] = tuple([gd.m_star * gd.n for gd in self.grid_disc])
        self.n_panels_tot: int = sum(self.n_bound_panels) + sum(self.n_wake_panels)

        # placeholder for aerodynamic local grid coordinates, and global coordinates for wing and wake
        self._x0_b: Optional[ArrayList] = None
        self._zeta0_b: Optional[ArrayList] = None
        self._zeta0_w: Optional[ArrayList] = None

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
        self._delta_w: Optional[list[Optional[Array]]] = None

        # kernel definitions per surface (seperate for wing and wake)
        if kernel is None:
            kernel = biot_savart_epsilon
            # kernel = biot_savart
        self.kernels_b: Sequence[KernelFunction] = self.n_surf * [kernel]
        self.kernels_w: Sequence[KernelFunction] = self.n_surf * [kernel]

        # surface names used for plotting
        self.surf_b_names: list[str] = [f"surf_{i}_bound" for i in range(self.n_surf)]
        self.surf_w_names: list[str] = [f"surf_{i}_wake" for i in range(self.n_surf)]

        # placeholder for time domain case
        self.zeta_b: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc])
        self.zeta_b_dot: ArrayList = ArrayList([
            jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc
        ])
        self.zeta_w: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m_star + 1, gd.n + 1, 3)) for gd in self.grid_disc])
        self.gamma_b: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc])
        self.gamma_b_dot: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc])
        self.gamma_w: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m_star, gd.n)) for gd in self.grid_disc])
        self.f_steady: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc])
        self.f_unsteady: ArrayList = ArrayList([jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc])
        self.n_tstep_tot: int = n_tstep
        self.t: Array = jnp.zeros(n_tstep)

        self._last_ts: int = -1  # used to store the last step where a solution was computed

    @property
    def flowfield(self) -> FlowField:
        if self._flowfield is None:
            raise ValueError("FlowField has not been set.")
        return self._flowfield

    @property
    def x0_b(self) -> ArrayList:
        if self._x0_b is None:
            raise ValueError("Design variable x0_b has not been set.")
        return self._x0_b

    @property
    def zeta0_b(self) -> ArrayList:
        if self._zeta0_b is None:
            raise ValueError("Design variable zeta0_b has not been set.")
        return self._zeta0_b

    @property
    def zeta0_w(self) -> ArrayList:
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

    def linearise(self,
                  i_ts: int,
                  wake_type: LinearWakeType = LinearWakeType.FREE,
                  bound_upwash: bool = True,
                  wake_upwash: bool = True,
                  unsteady_force: bool = True,
                  gamma_dot_state: bool = True) -> LinearAero:
        r"""
        Create linearised aerodynamic model at specified time step
         0: steady horseshoe wake
         1: prescribed wake (from nonlinear case)
         2: free wake (from nonlinear case)
        :param i_ts: Time step index to linearise about
        :param wake_type: Type of wake model to use in linearisation (frozen, prescribed, or free)
        :param bound_upwash: If true, linearise for flowfield pertubations at the bound vortex vertex
        :param wake_upwash: If true, linearise for flowfield pertubations at the wake vortex vertex
        :param unsteady_force: If true, include unsteady force terms in linearisation
        :param gamma_dot_state: If true, include gamma dot as a state in the linear model
        :return: LinearAero model linearised at specified time step
        """
        return LinearAero(self,
                          self[i_ts],
                          wake_type=wake_type,
                          bound_upwash=bound_upwash,
                          wake_upwash=wake_upwash,
                            unsteady_force=unsteady_force,
                          gamma_dot_state=gamma_dot_state)

    def set_design_variables(self,
                             dt: float | Array,
                             flowfield: FlowField,
                             delta_w: Optional[Sequence[Array] | Array],
                             x0_aero: ArrayList | Sequence[Array] | Array,
                             hg0: Array) -> None:
        r"""
        Set aerodynamic design variables for solution.
        :param dt: Time step length
        :param flowfield: FlowField object defining the background flow in space and time
        :param delta_w: Vector to define segment lengths of a variable wake discretisation per surface. If None, this
        will use a uniform discretisation, as in the canonical UVLM.
        :param x0_aero: Aerodynamic local grid coordinates, [n_surf][m+1, n+1, 3]
        :param hg0: Beam global grid coordinates, [n, 4, 4]
        """
        if isinstance(delta_w, Array):
            delta_w = [delta_w]
        if isinstance(x0_aero, Array):
            x0_aero = [x0_aero]
        if isinstance(x0_aero, Sequence):
            x0_aero = ArrayList(x0_aero)

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
                if dw_ is None:
                    self._delta_w.append(None)
                else:
                    check_arr_shape(dw_, (self.grid_disc[i_surf].m_star, ), "delta_w")
                    self._delta_w.append(dw_)
        else:
            # auto compute delta_w based on freestream and dt
            self._delta_w = self.n_surf * [None]
        self.initialise_wake()

    @replace_self
    def extend_n_tstep(self, n_tstep: int) -> Self:
        r"""
        Extend the time domain arrays in the case object by the specified number of time steps
        :param n_tstep: Number of time steps to extend by
        """
        for var in (self.zeta_b, self.zeta_b_dot, self.zeta_w, self.gamma_b, self.gamma_b_dot, self.gamma_w,
                    self.f_steady, self.f_unsteady, self.t):
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
            zeta_b=ArrayList([self.zeta_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            zeta_b_dot=ArrayList([self.zeta_b_dot[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            zeta_w=ArrayList([self.zeta_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            gamma_b=ArrayList([self.gamma_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            gamma_b_dot=ArrayList([self.gamma_b_dot[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            gamma_w=ArrayList([self.gamma_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            f_steady=ArrayList([self.f_steady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            f_unsteady=ArrayList([self.f_unsteady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]),
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t[i_ts],
            n_surf=self.n_surf
        )

    def get_v_ind(self, x: Array, i_ts: int) -> Array:
        """
        Compute the induced velocity at points x due to a system of vortex elements.
        :param x: Points at which to compute induced velocity, [..., 3]
        :param i_ts: Timestep index
        :return: Induced velocity at points x, of shapes [..., 3].
        """

        gamma_flat = self.get_gamma_vect(i_ts)

        aic = compute_aic_sys_assembled(
            [x],
            self.get_zeta(i_ts),
            [*self.kernels_b, *self.kernels_w],
        )  # shapes [n_x, n_gamma, 3]

        return jnp.einsum("ijk,j->ik", aic, gamma_flat).reshape(x.shape)

    def get_v_background(self, i_ts: int, x: Array) -> Array:
        r"""
        Get background (freestream) velocity at specified points and time step
        :param i_ts: Timestep index
        :param x: Points to evaluate background velocity at, [..., 3]
        :return: Background velocity at points, [..., 3]
        """
        return self.flowfield.vmap_call(x.reshape(-1, 3), self.t[i_ts]).reshape(x.shape)


    def get_v_tot(self, i_ts: int, x: Array) -> Array:
        r"""
        Get induced velocity at specified points and time step by the elements in the flow and the freesteam
        :param i_ts: Timestep index
        :param x: Points to evaluate induced velocity at, [..., 3]
        :return: Induced velocity at points, [..., 3]
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

    def get_zeta_b(self, i_ts: int) -> ArrayList:
        r"""
        Get bound grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        """
        return self.zeta_b.index_all((i_ts, ...))

    def get_zeta_dot_b(self, i_ts: int) -> ArrayList:
        r"""
        Get bound grid velocities for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        """
        return self.zeta_b_dot.index_all((i_ts, ...))

    def get_zeta_w(self, i_ts: int) -> ArrayList:
        r"""
        Get wake grid coordinates for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        """
        return self.zeta_w.index_all((i_ts, ...))

    def get_zeta(self, i_ts: int) -> ArrayList:
        r"""
        Get full grid coordinates (bound, wake) for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: List of grid coordinates for each surface, [2 * n_surf][m_, n_, 3]
        """
        return self.get_zeta_b(i_ts).combine(self.get_zeta_w(i_ts))

    def hg_to_zeta(self, hg: Array) -> ArrayList:
        zetas = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg = jnp.take(hg, self.dof_mapping[i_surf], axis=0)   # [n, 4, 4]

            zetas.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg, self.x0_b[i_surf]
                )
            )
        return zetas

    def hg_dot_to_zeta_dot(self, hg_dot: Array) -> ArrayList:
        zeta_dots = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg_dot = jnp.take(hg_dot, self.dof_mapping[i_surf], axis=0)   # [n, 4, 4]
            zeta_dots.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg_dot, self.x0_b[i_surf]
                )
            )
        return zeta_dots


    def get_gamma_w(self, i_ts: int) -> ArrayList:
        return self.gamma_w.index_all((i_ts, ...))

    def get_gamma_w_vect(self, i_ts: int) -> Array:
        r"""
        Get wake circulation strengths vector for all surfaces at specified time step
        :param i_ts: Timestep index
        :return: Wake circulation strengths vector, [gamma_w_tot]
        """
        return self.get_gamma_w(i_ts).flatten()

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

    def get_gamma_b(self, i_ts: int) -> ArrayList:
        return self.gamma_b.index_all((i_ts, ...))

    def get_gamma_b_vect(self, i_ts: int) -> Array:
        return self.get_gamma_b(i_ts).flatten()

    @replace_self
    def set_zeta_b(self, zeta_b: ArrayList, i_ts: int) -> Self:
        r"""
        Set bound grid coordinates from list of grid coordinates at specified time step
        :param zeta_b: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b[i_surf] = self.zeta_b[i_surf].at[i_ts, ...].set(zeta_b[i_surf])
        return self

    @replace_self
    def set_zeta_b_dot(self, zeta_b_dot: ArrayList, i_ts: int) -> Self:
        r"""
        Set bound grid velocities from list of grid coordinates at specified time step
        :param zeta_b_dot: List of bound grid coordinates for each surface, [n_surf][m+1, n+1, 3]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            self.zeta_b_dot[i_surf] = self.zeta_b_dot[i_surf].at[i_ts, ...].set(zeta_b_dot[i_surf])
        return self

    @replace_self
    def set_zeta_w(self, zeta_w: ArrayList, i_ts: int) -> Self:
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
        zeta_te = self.get_zeta_te_surf(i_ts, i_surf)  # [zeta_n, 3]
        if self.grid_disc[i_surf].m_star == 0:
            warn("Horseshoe wake requested but m_star == 0, skipping.")
            return zeta_te[None, :]
        else:

            wake_end = zeta_te + (self.flowfield.u_inf_dir * horseshoe_length)[None, :] # [zeta_n, 3]
            return jnp.stack((zeta_te, wake_end), axis=0)

    @replace_self
    def initialise_wake(self) -> Self:
        r"""
        Generate initial wake grid coordinates
        Note that the wake grid does not include the trailing edge, assuming that delta_w[0, :] != 0
        """

        self._zeta0_w = ArrayList([])
        for i_surf, this_delta_w in enumerate(self.delta_w):
            # get bound grid coordinates
            zeta_te = self.zeta0_b[i_surf][-1, :, :]   # [n+1, 3]

            # set wake grid coordinates as trailing edge + displacement
            if this_delta_w is None:
                this_delta_w = jnp.ones(self.grid_disc[i_surf].m_star) * self.dt * self.flowfield.u_inf_mag
            grid_s = jnp.concatenate((jnp.zeros(1), jnp.cumsum(this_delta_w)))

            self._zeta0_w.append(zeta_te[None, :, :] + jnp.outer(grid_s, self.flowfield.u_inf_dir)[:, None, :])
        return self


    @replace_self
    def calculate_steady_forcing(self, i_ts: int) -> Self:
        f_steady = steady_forcing(self.get_zeta_b(i_ts),
                           self.get_zeta_dot_b(i_ts),
                           self.get_gamma_b(i_ts),
                           self.get_gamma_w(i_ts),
                           lambda x_: self.get_v_tot(i_ts, x_),
                            None,
                           self.flowfield.rho)
        for i_surf in range(self.n_surf):
            self.f_steady[i_surf] = self.f_steady[i_surf].at[i_ts, ...].set(f_steady[i_surf])
        return self

    @replace_self
    def calculate_gamma_dot(self, i_ts: int) -> Self:
        for i_surf in range(self.n_surf):
            self.gamma_b_dot[i_surf] = self.gamma_b_dot[i_surf].at[i_ts, ...].set(
                finite_difference(i_ts, self.gamma_b[i_surf], self.dt, 0, order=1)
            )
        return self

    @replace_self
    def calculate_unsteady_forcing(self, i_ts: int, ncs: Sequence[Array]) -> Self:
        for i_surf in range(self.n_surf):
            self.f_unsteady[i_surf] = self.f_unsteady[i_surf].at[i_ts, ...].set(
                self.calculate_surf_unsteady_forcing(i_ts, i_surf, ncs[i_surf])
            )
        return self

    def calculate_surf_unsteady_forcing(self, i_ts: int, i_surf: int, nc: Array) -> Array:
        return split_to_vertex(self.flowfield.rho * self.gamma_b_dot[i_surf][i_ts, ..., None] * nc, (0, 1)) # [gamma_m+1, gamma_n+1, 3]

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

        # set the current time step
        if static:
            self.t = self.t.at[i_ts].set(jax.lax.select(i_ts, self.t[i_ts - 1], 0.0))
        else:
            self.t = self.t.at[i_ts].set(jax.lax.select(i_ts, self.t[i_ts - 1] + self.dt, 0.0))

        zetas_b = self.zeta0_b if hg is None else self.hg_to_zeta(hg)
        self.set_zeta_b(zetas_b, i_ts)

        cs = get_c(zetas_b)
        ncs = get_nc(zetas_b)

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
            def v_wake_prop(x_: Array) -> Array:
                if free_wake:
                    return self.get_v_tot(i_ts - 1, x_)
                else:
                    return self.get_v_background(i_ts - 1, x_)

            # propagate wake
            zeta_ws, gamma_ws = propagate_wake(self.get_gamma_b(i_ts - 1),
                                               self.get_gamma_w(i_ts - 1),
                                               zetas_b,
                                               self.get_zeta_w(i_ts - 1),
                                               self.delta_w,
                                               v_wake_prop,
                                               self.dt,
                                               frozen_wake=False
                                               )

            self.set_gamma_w(gamma_ws, i_ts)
            self.set_zeta_w(zeta_ws, i_ts)

        # set wake grid coordinates
        self.set_zeta_w(zeta_ws, i_ts) if not horseshoe else self.set_zeta_w(self.zeta0_w, i_ts)

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

        n_vect = jnp.concatenate([nc.reshape(-1, 3) for nc in ncs], axis=0)
        v_bc_n = jnp.einsum('ij,ij->i', v_bc, n_vect)

        if not static:
            aic_w = assemble_aic_sys(aic_w_blocks)
            v_bc_n -= aic_w @ self.get_gamma_w_vect(i_ts)

        gamma_b_vect: Array = jnp.linalg.solve(aic, v_bc_n)
        self.set_gamma_b(gamma_b_vect, i_ts)

        if static:
            self.set_gamma_w_static(i_ts)
        else:
            self.calculate_gamma_dot(i_ts)
            self.calculate_unsteady_forcing(i_ts, ncs)
        self.calculate_steady_forcing(i_ts)

        return self

    @replace_self
    @print_with_time("Computing static solution...", "Static solution complete in {:.2f} seconds.")
    # TODO: add free wake
    def solve_static(self,
                     i_ts: Optional[int] = None,
                     hg: Optional[Array] = None,
                     horseshoe: bool = False) -> Self:
        if i_ts is None:
            i_ts = self._last_ts + 1
        self._last_ts = i_ts    # update last solved timestep

        self.solve(i_ts,
                   hg, None, static=True, free_wake=False, horseshoe=horseshoe)

        return self

    @replace_self
    @print_with_time(
        "Computing prescribed dynamic solution...",
        "Prescribed dynamic solution complete in {:.2f} seconds."
    )
    def solve_prescribed_dynamic(self,
                                 hg_t: Array,
                                 hg_dot_t: Array,
                                 free_wake: bool = False,
                                 i_ts_start: Optional[int] = None) -> Self:

        check_arr_shape(hg_t, (None, None, 4, 4), "hg")
        check_if_all_se3_g(hg_t, True)
        if hg_dot_t is not None:
            if hg_t.shape != hg_dot_t.shape:
                raise ValueError(
                    f"hg_dot must have the same shapes as hg, got {hg_dot_t.shape} vs {hg_t.shape}"
                )
        check_if_all_se3_a(hg_dot_t, True)

        n_tstep = hg_t.shape[0]

        if i_ts_start is None:
            i_ts_start = self._last_ts + 1
        self._last_ts = i_ts_start + n_tstep    # update last solved timestep

        def step_func(i_ts_: int, case: AeroCase) -> AeroCase:
            case.solve(
                i_ts_,
                hg_t[i_ts_, ...],
                hg_dot_t[i_ts_, ...],
                static=False,
                free_wake=free_wake,
                horseshoe=False,
            )
            jax_print("UVLM timestep {i_ts_}", i_ts_=i_ts_)
            return case

        obj = fori_loop(
            i_ts_start,
            i_ts_start + n_tstep,
            step_func,
            init_val=self,
        )
        jax.block_until_ready(obj)
        return obj

    @print_with_time(
        "Plotting aerodynamic grid...",
        "Aerodynamic grid plotted in {:.2f} seconds.",
    )
    def plot(self, directory: PathLike, index: Optional[slice | Sequence[int] | int | Array] = None, plot_wake: bool = True) -> None:
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep_tot)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index, )
        elif index is None:
            index_ = jnp.arange(self.n_tstep_tot)
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
                name = ((self.surf_b_names + self.surf_w_names)[i_surf] + "_ts")
                write_pvd(directory, name, surf_paths, list(self.t[index_]))
            except IndexError:
                pass

    def reference_snapshot(self) -> AeroSnapshot:
        return AeroSnapshot(
            zeta_b=self.zeta0_b,
            zeta_b_dot=ArrayList([jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]),
            zeta_w=self.zeta0_w,
            gamma_b=ArrayList([jnp.zeros((gd.m, gd.n)) for gd in self.grid_disc]),
            gamma_b_dot=ArrayList([jnp.zeros((gd.m, gd.n)) for gd in self.grid_disc]),
            gamma_w=ArrayList([jnp.zeros((gd.m_star, gd.n)) for gd in self.grid_disc]),
            f_steady=ArrayList([jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]),
            f_unsteady=ArrayList([jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]),
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=-1,
            t=jnp.zeros(()),
            n_surf=self.n_surf
        )

    @print_with_time(
        "Plotting reference aerodynamic grid...",
        "Reference aerodynamic grid plotted in {:.2f} seconds.",
    )
    def plot_reference(self, directory: PathLike, plot_wake: bool = True) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(Path(directory).resolve(), plot_wake=plot_wake)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return (
            "n_surf",
            "grid_disc",
            "n_bound_panels",
            "n_wake_panels",
            "n_panels_tot",
            "gamma_b_slice",
            "gamma_w_slice",
            "dof_mapping",
            "variable_wake_disc",
            "kernels_b",
            "kernels_w",
            "surf_b_names",
            "surf_w_names",
        )

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "_x0_b",
            "_zeta0_b",
            "_zeta0_w",
            "_dt",
            "_flowfield",
            "_delta_w",
            "zeta_b",
            "zeta_b_dot",
            "zeta_w",
            "gamma_b",
            "gamma_b_dot",
            "gamma_w",
            "f_steady",
            "f_unsteady",
            "n_tstep_tot",
            "t",
            "_last_ts",
        )
