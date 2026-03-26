from __future__ import annotations
from collections.abc import Sequence
from typing import Optional, TYPE_CHECKING
import os
from pathlib import Path
from functools import singledispatchmethod

import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import fori_loop

from aegrad.utils import _make_pytree
from aegrad.algebra.test_routines import check_if_all_se3_g, check_if_all_se3_a
from aegrad.aero.utils import propagate_wake, compute_c, compute_nc
from aegrad.constants import HORSESHOE_LENGTH
from aegrad.algebra.array_utils import (
    check_arr_dtype,
    neighbour_average,
    check_arr_shape,
    ArrayList,
)
from aegrad.aero.data_structures import (
    GridDiscretization,
    DynamicAeroCase,
    AeroSnapshot,
)
from aegrad.aero.flowfields import FlowField
from aegrad.aero.utils import KernelFunction, biot_savart_epsilon

from aegrad.algebra.se3 import vect_product as se3_vect_product
from aero.aic import compute_v_ind, compute_aic_solve

if TYPE_CHECKING:
    from aegrad.aero.linear.linear_uvlm import LinearUVLM, LinearWakeType
from aegrad.print_utils import warn, jax_print


@_make_pytree
class UVLM:
    r"""
    Class to define an unsteady vortex lattice method aerodynamic case with arbitrary number of aerodynamic surfaces.
    """

    def __init__(
        self,
        grid_shapes: Sequence[GridDiscretization | tuple[int, int, int]],
        dof_mapping: ArrayList | Sequence[Array] | Array,
        variable_wake_disc: bool = False,
        mirror_point: Optional[Array] = None,
        mirror_normal: Optional[Array] = None,
        kernel: Optional[KernelFunction] = None,
    ) -> None:
        r"""
        Initialise UVLM class with all non-design parameters
        :param grid_shapes: Discretisation(s) for the number of chordwise, spanwise and wake-wise panels for each surface.
        May be passed using either the GridDiscretization class or a tuple of integers (m, varphi, m_star). Multiple surfaces
        may be defined by passing a sequence of GridDiscretization instances or tuples.
        :param dof_mapping: Mapping from aerodynamic grid points to structure_dv grid points for each surface.
        :param variable_wake_disc: If true, allow for variable wake discretisation per surface.
        :param mirror_point: Optional point in mirror plane. If provided, this will apply mirroring of the aerodynamic
        geometry and flow about the plane defined by this point and the mirror varphi, [3].
        :param mirror_normal: Optional varphi vector for mirror plane, [3].
        :param kernel: Input for custom kernel function to use for induced velocity calculations.
        """

        # case for single inputs
        if isinstance(dof_mapping, Array):
            dof_mapping = [dof_mapping]

        # number of aerodynamic surfaces
        self.n_surf: int = len(grid_shapes)

        # set grid discretizations parameters for number of panels
        grid_disc = []

        for grid in grid_shapes:
            if isinstance(grid, Sequence):
                if len(grid) != 3:
                    raise ValueError(
                        "Grid arr_list_shapes tuple must have exactly three elements (m, varphi, m_star)"
                    )
                grid_disc.append(GridDiscretization(*grid))
            elif isinstance(grid, GridDiscretization):
                grid_disc.append(grid)
            else:
                raise TypeError(
                    "Grid arr_list_shapes must be either a Sequence of three integers or a GridDiscretization instance"
                )
        self.grid_disc: tuple[GridDiscretization] = tuple(grid_disc)

        # count of number of panels
        self.n_bound_panels: tuple[int, ...] = tuple(
            [gd.m * gd.n for gd in self.grid_disc]
        )
        self.n_wake_panels: tuple[int, ...] = tuple(
            [gd.m_star * gd.n for gd in self.grid_disc]
        )
        self.n_panels_tot: int = sum(self.n_bound_panels) + sum(self.n_wake_panels)

        # placeholder for aerodynamic local grid coordinates, and global coordinates for wing and wake
        self._x0_b: Optional[ArrayList] = None
        self._zeta0_b: Optional[ArrayList] = None
        self._zeta0_w: Optional[ArrayList] = None

        self.gamma_b_slice, self.gamma_w_slice = self._make_gamma_slices()

        # store DOF mapping
        if len(dof_mapping) != self.n_surf:
            raise ValueError(
                f"Expected {self.n_surf} DOF mapping arrays, got {len(dof_mapping)}"
            )
        for i_surf, map_ in enumerate(dof_mapping):
            check_arr_dtype(map_, int, "dof_mapping")
            check_arr_shape(map_, (self.grid_disc[i_surf].n + 1,), "grid_disc")

        self.dof_mapping: ArrayList = ArrayList(dof_mapping)

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

        # mirror definitions
        if (mirror_point is None and mirror_normal is not None) or (
            mirror_point is not None and mirror_normal is None
        ):
            raise ValueError(
                "Both mirror_point and mirror_normal must be provided to apply mirroring, or both must be None to apply no mirroring."
            )

        if mirror_point is None:
            self.mirror_point: Optional[Array] = None
            self.mirror_normal: Optional[Array] = None
        else:
            self.mirror_point = mirror_point
            self.mirror_normal = mirror_normal / jnp.linalg.norm(
                mirror_normal
            )  # normalise

        # surface names used for plotting
        self.surf_b_names: list[str] = [f"surf_{i}_bound" for i in range(self.n_surf)]
        self.surf_w_names: list[str] = [f"surf_{i}_wake" for i in range(self.n_surf)]

    @property
    def flowfield(self) -> FlowField:
        r"""
        Get the FlowField object defining the background flow in space and time
        :return: FlowField object
        """
        if self._flowfield is None:
            raise ValueError("FlowField has not been set.")
        return self._flowfield

    @property
    def x0_b(self) -> ArrayList:
        r"""
        Get the aerodynamic local grid coordinates at the reference configuration
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        if self._x0_b is None:
            raise ValueError("Design variable x0_b has not been set.")
        return self._x0_b

    @property
    def zeta0_b(self) -> ArrayList:
        r"""
        Get the aerodynamic global grid coordinates at the reference configuration
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        if self._zeta0_b is None:
            raise ValueError("Design variable zeta0_b has not been set.")
        return self._zeta0_b

    @property
    def zeta0_w(self) -> ArrayList:
        r"""
        Get the aerodynamic global wake grid coordinates at the reference configuration
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m_star, zeta_n, 3]
        """
        if self._zeta0_w is None:
            raise ValueError("Design variable zeta0_w has not been set.")
        return self._zeta0_w

    @property
    def dt(self) -> Array:
        r"""
        Get the time step length
        :return: Time step length
        """
        if self._dt is None:
            raise ValueError("Time step length dt has not been set.")
        return self._dt

    @property
    def delta_w(self) -> Sequence[Array]:
        r"""
        Get the wake displacement vector defining segment lengths of a variable wake discretisation per surface.
        :return: Wake displacement vector(s) for each surface, [n_surf][m_star]
        """
        if self._delta_w is None:
            raise ValueError("Wake displacement delta_w has not been set.")
        return self._delta_w

    def linearise(
        self,
        reference: AeroSnapshot,
        wake_type: Optional[LinearWakeType] = None,
        bound_upwash: bool = True,
        wake_upwash: bool = True,
        unsteady_force: bool = True,
        gamma_dot_state: bool = True,
    ) -> LinearUVLM:
        r"""
        Create linearised aerodynamic model.
        :param reference: Reference StaticAero around which to linearise.
        :param wake_type: Type of wake model to use in linearisation, with options given from the LinearWakeType class
         (frozen, prescribed, or free). Value of None defaults to prescribed.
        :param bound_upwash: If true, linearise for flowfield perturbations at the bound vortex vertex.
        :param wake_upwash: If true, linearise for flowfield perturbations at the wake vortex vertex.
        :param unsteady_force: If true, include unsteady force terms in linearisation.
        :param gamma_dot_state: If true, include time derivative of bound circulation as a state in the linear model.
        :return: LinearUVLM model linearised at specified time step.
        """

        # local import used to prevent circular import issues
        from aegrad.aero.linear.linear_uvlm import LinearUVLM, LinearWakeType

        if wake_type is None:
            wake_type = LinearWakeType.PRESCRIBED

        return LinearUVLM(
            self,
            reference=reference,
            wake_type=wake_type,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
            gamma_dot_state=gamma_dot_state,
        )

    def set_design_variables(
        self,
        dt: float | Array,
        flowfield: FlowField,
        delta_w: Optional[Sequence[Array] | Array],
        x0_aero: ArrayList | Sequence[Array] | Array,
        hg0: Array,
    ) -> None:
        r"""
        Set aerodynamic design variables for solution.
        :param dt: Time step length
        :param flowfield: FlowField object defining the background flow in space and time
        :param delta_w: Vector to define segment lengths of a variable wake discretisation per surface. If None, this
        will use a uniform discretisation, as in the canonical UVLM.
        :param x0_aero: Aerodynamic local grid coordinates, [n_surf][zeta_m, zeta_n, 3]
        :param hg0: Beam reference global grid coordinates, [zeta_n, 4, 4]
        """
        if isinstance(delta_w, Array):
            delta_w = [delta_w]
        if isinstance(x0_aero, Array):
            x0_aero = [x0_aero]
        if isinstance(x0_aero, Sequence):
            x0_aero = ArrayList(x0_aero)

        # set aerodynamic local coordinates
        if len(x0_aero) != self.n_surf:
            raise ValueError(
                f"Expected {self.n_surf} aerodynamic grid coordinate arrays, got {len(x0_aero)}"
            )

        for i_surf in range(self.n_surf):
            check_arr_shape(
                x0_aero[i_surf],
                (self.grid_disc[i_surf].m + 1, self.grid_disc[i_surf].n + 1, 3),
                "x0_aero",
            )
        self._x0_b = x0_aero

        # set global grid coordinates for bound and wake
        check_arr_shape(hg0, (None, 4, 4), "hg0")
        self._zeta0_b = self._hg_to_zeta(hg0)

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
                    check_arr_shape(dw_, (self.grid_disc[i_surf].m_star,), "delta_w")
                    self._delta_w.append(dw_)
        else:
            # auto compute delta_w based on freestream and dt
            self._delta_w = self.n_surf * [None]
        self._zeta0_w = self.initialise_wake()

    def _hg_to_zeta(self, hg: Array) -> ArrayList:
        r"""
        Convert beam global grid coordinates to aerodynamic global grid coordinates.
        :param hg: Beam global grid coordinates, [zeta_n, 4, 4]
        :return: Full aerodynamic global grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        zetas = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg = jnp.take(hg, self.dof_mapping[i_surf], axis=0)  # [varphi, 4, 4]

            zetas.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg, self.x0_b[i_surf]
                )
            )
        return zetas

    def _hg_dot_to_zeta_dot(self, hg_dot: Array) -> ArrayList:
        r"""
        Convert beam global grid velocities to aerodynamic global grid velocities.
        :param hg_dot: Beam global grid velocities, [zeta_n, 4, 4]
        :return: Full aerodynamic global grid velocites for each surface, [n_surf][zeta_m, zeta_n, 3]
        """
        zeta_dots = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg_dot = jnp.take(
                hg_dot, self.dof_mapping[i_surf], axis=0
            )  # [varphi, 4, 4]
            zeta_dots.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg_dot, self.x0_b[i_surf]
                )
            )
        return zeta_dots

    def _make_gamma_slices(self) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        r"""
        Create slices for the bound and wake circulation strengths in their respective solution vectors based on the grid
        discretizations.
        :return: Sequence of slices for bound and wake circulation strengths.
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

    def _make_surf_horseshoe_wake(
        self, case: DynamicAeroCase, i_ts: int, i_surf: int, horseshoe_length: float
    ) -> Array:
        r"""
        Create a horseshoe wake grid for a given surface at a given time step.
        :param i_ts: Timestep index
        :param i_surf: Surface index
        :param horseshoe_length: Length of horseshoe wake to generate
        :return: Wake grid coordinates, [2, zeta_n, 3]
        """
        zeta_te = case.zeta_b[i_surf][i_ts, -1, ...]  # [zeta_n, 3]
        if self.grid_disc[i_surf].m_star == 0:
            warn("Horseshoe wake requested but m_star == 0, skipping.")
            return zeta_te[None, :]
        else:
            wake_end = (
                zeta_te + (self.flowfield.u_inf_dir * horseshoe_length)[None, :]
            )  # [zeta_n, 3]
            return jnp.stack((zeta_te, wake_end), axis=0)

    def initialise_wake(self, zeta_b: Optional[ArrayList] = None) -> ArrayList:
        r"""
        Generate initial wake grid coordinates, based on the bound grid coordinates and the freestream conditions.
        :param zeta_b: Initial wake grid coordinates, [n_surf][zeta_m, zeta_n, 3]. If None, this will use the
        initialised bound grid coordinates based on hg0.
        :return: Initial wake grid coordinates, [zeta_m_star, zeta_n, 3]
        """
        if zeta_b is None:
            zeta_b = self.zeta0_b

        zeta0_w = ArrayList([])
        for i_surf, this_delta_w in enumerate(self.delta_w):
            # get bound grid coordinates
            zeta_te = zeta_b[i_surf][-1, :, :]  # [varphi+1, 3]

            # set wake grid coordinates as trailing edge + displacement
            if this_delta_w is None:
                this_delta_w = (
                    jnp.ones(self.grid_disc[i_surf].m_star)
                    * self.dt
                    * self.flowfield.u_inf_mag
                )
            grid_s = jnp.concatenate((jnp.zeros(1), jnp.cumsum(this_delta_w)))

            zeta0_w.append(
                zeta_te[None, :, :]
                + jnp.outer(grid_s, self.flowfield.u_inf_dir)[:, None, :]
            )
        return zeta0_w

    def set_gamma_b(self, case: DynamicAeroCase, gamma_vec: Array, i_ts: int) -> None:
        r"""
        Set bound circulation strengths from total circulation strengths vector at specified time step
        :param case: DynamicAeroCase object
        :param gamma_vec: Total circulation strengths vector, [gamma_b_tot]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            case.gamma_b[i_surf] = (
                case.gamma_b[i_surf]
                .at[i_ts, ...]
                .set(
                    gamma_vec[self.gamma_b_slice[i_surf]].reshape(
                        self.grid_disc[i_surf].m, self.grid_disc[i_surf].n
                    )
                )
            )

    @singledispatchmethod
    def set_gamma_w(self, gamma_vec: Array, case: DynamicAeroCase, i_ts: int) -> None:
        r"""
        Set wake circulation strengths from total circulation strengths at specified time step. Can be passed either a
        full vector of strengths, or a sequence of strengths per surface.
        :param case: DynamicAeroCase object
        :param gamma_vec: Total circulation strengths vector, [gamma_w_tot]
        :param i_ts: Timestep index
        """
        for i_surf in range(self.n_surf):
            case.gamma_w[i_surf] = (
                case.gamma_w[i_surf]
                .at[i_ts, ...]
                .set(
                    gamma_vec[self.gamma_w_slice[i_surf]].reshape(
                        self.grid_disc[i_surf].m_star, self.grid_disc[i_surf].n
                    )
                )
            )

    @set_gamma_w.register(Sequence)
    def _(
        self, gamma_list: Sequence[Array] | ArrayList, case: DynamicAeroCase, i_ts: int
    ) -> None:
        for i_surf in range(self.n_surf):
            case.gamma_w[i_surf] = (
                case.gamma_w[i_surf].at[i_ts, ...].set(gamma_list[i_surf])
            )

    def _solve(
        self,
        case: DynamicAeroCase,
        i_ts: int,
        hg: Optional[Array],
        hg_dot: Optional[Array],
        static: bool,
        free_wake: bool,
        horseshoe: bool,
    ) -> DynamicAeroCase:
        r"""
        Solve the UVLM equations for a single time step. Can be used for both static and dynamic solves.
        :param case: DynamicAeroCase object containing solution at previous time steps
        :param i_ts: Timestep index to solve for
        :param hg: Beam global grid coordinates, [zeta_n, 4, 4]
        :param hg_dot: Beam global grid velocities, [zeta_n, 4, 4]
        :param static: If true, perform a static solve
        :param free_wake: If true, use free wake propagation in dynamic solve
        :param horseshoe: If true, replace the wake with a horseshoe wake in static solve which extends a fixed distance
        """
        if not static and horseshoe:
            warn(
                "Horseshoe wake not compatible with non-static solve. Overriding horseshoe to False."
            )
            horseshoe = False

        # set the current time step
        if not static:
            case.t = case.t.at[i_ts].set(
                jax.lax.select(i_ts, case.t[i_ts - 1] + self.dt, 0.0)
            )

        zetas_b = self.zeta0_b if hg is None else self._hg_to_zeta(hg)
        case.set_arraylist_at_ts("_zeta_b", values=zetas_b, i_ts=i_ts)
        case.compute_c(i_ts=i_ts)
        case.compute_nc(i_ts=i_ts)

        if hg_dot is None:
            c_dot = None
        else:
            zeta_b_dot = self._hg_dot_to_zeta_dot(hg_dot)

            case.set_arraylist_at_ts("_zeta_b_dot", values=zeta_b_dot, i_ts=i_ts)

            c_dot = ArrayList(
                [neighbour_average(zeta_dot, axes=(0, 1)) for zeta_dot in zeta_b_dot]
            )

        if static:
            # initialise wake
            # update zeta0_w
            if horseshoe:
                zeta_ws = [
                    self._make_surf_horseshoe_wake(case, i_ts, i_surf, HORSESHOE_LENGTH)
                    for i_surf in range(self.n_surf)
                ]
            else:
                # re-initialise wake. This is wasteful when there is no coupled structure_dv as zeta_ws will equal zeta0_w
                # but is necessary to update the wake grid coordinates when there is a coupled structure_dv as the bound
                # grid coordinates will have changed from the initial configuration.
                zeta_ws = self.initialise_wake(zetas_b)
        else:

            def v_wake_prop(x_: Array) -> Array:
                v = self.flowfield.vmap_call(x=x_, t=case.t[i_ts])
                if free_wake:
                    v += compute_v_ind(
                        cs=x_,
                        zetas=case.zeta_full(i_ts - 1),
                        gammas=case.gamma_full(i_ts - 1),
                        kernels=[*self.kernels_b, *self.kernels_w],
                    )
                return v

            # propagate wake
            zeta_ws, gamma_ws = propagate_wake(
                case.gamma_b.index_all(i_ts - 1, ...),
                case.gamma_w.index_all(i_ts - 1, ...),
                zetas_b,
                case.zeta_w.index_all(i_ts - 1, ...),
                self.delta_w,
                v_wake_prop,
                self.dt,
                frozen_wake=False,
            )

            self.set_gamma_w(gamma_ws, case, i_ts)
            case.set_arraylist_at_ts("_zeta_w", values=zeta_ws, i_ts=i_ts)

        # set wake grid coordinates. If using horseshoe, it will still create a regular wake for plotting
        case.set_arraylist_at_ts(
            "_zeta_w", values=zeta_ws, i_ts=i_ts
        ) if not horseshoe else case.set_arraylist_at_ts(
            "_zeta_w", values=self.initialise_wake(zetas_b), i_ts=i_ts
        )

        aic_solve = compute_aic_solve(
            cs=case.get_c(i_ts),
            ns=case.get_n(i_ts),
            zetas_b=zetas_b,
            zetas_w=zeta_ws if static else None,
            kernels_b=self.kernels_b,
            kernels_w=self.kernels_w if static else None,
            mirror_normal=self.mirror_normal,
            mirror_point=self.mirror_point,
        )

        v_bc = self.flowfield.surf_vmap_call(
            xs=case.get_c(i_ts=i_ts), t=case.t[i_ts]
        )  # [n_surf][m, varphi, 3]

        if not static:
            # strucural component
            v_bc -= c_dot

            # find wake component
            v_bc += compute_v_ind(
                cs=case.get_c(i_ts),
                zetas=case.zeta_w.index_all(i_ts, ...),
                gammas=case.gamma_w.index_all(i_ts, ...),
                kernels=self.kernels_w,
            )

        v_bc_n = ArrayList.einsum(
            "ijk,ijk->ij", v_bc, case.nc.index_all(i_ts, ...)
        )  # [c_tot]

        gamma_b_vect = jnp.linalg.solve(aic_solve, -v_bc_n.flatten())
        self.set_gamma_b(case, gamma_b_vect, i_ts)

        if static:
            case.set_gamma_w_static(i_ts)
        else:
            case.compute_gamma_dot(i_ts, self.dt)
            case.calculate_unsteady_forcing(i_ts=i_ts)
        case.calculate_steady_forcing(i_ts=i_ts)

        return case

    def initialise_case_object(self, n_tstep: int, horseshoe: bool) -> DynamicAeroCase:
        r"""
        Initialise an DynamicAeroCase object to store the solution of the aerodynamic case, with the correct dimensions and
        initial conditions.
        :param n_tstep: Number of time steps to solve for in dynamic solution
        :param horseshoe: Whether to use horseshoe or not
        :return: DynamicAeroCase object with initial conditions set
        """
        # zero initialise
        return DynamicAeroCase(
            zeta_b=ArrayList(
                [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            zeta_b_dot=ArrayList(
                [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            zeta_w=ArrayList(
                [
                    jnp.zeros((n_tstep, gd.m_star + 1, gd.n + 1, 3))
                    for gd in self.grid_disc
                ]
            ),
            gamma_b=ArrayList(
                [jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc]
            ),
            gamma_b_dot=ArrayList(
                [jnp.zeros((n_tstep, gd.m, gd.n)) for gd in self.grid_disc]
            ),
            gamma_w=ArrayList(
                [jnp.zeros((n_tstep, gd.m_star, gd.n)) for gd in self.grid_disc]
            ),
            f_steady=ArrayList(
                [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            f_unsteady=ArrayList(
                [jnp.zeros((n_tstep, gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            c=ArrayList([jnp.zeros((n_tstep, gd.m, gd.n, 3)) for gd in self.grid_disc]),
            nc=ArrayList(
                [jnp.zeros((n_tstep, gd.m, gd.n, 3)) for gd in self.grid_disc]
            ),
            kernels=[*self.kernels_b, *self.kernels_w],
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=jnp.arange(n_tstep),
            t=jnp.zeros(n_tstep),
            dof_mapping=self.dof_mapping,
            horseshoe=horseshoe,
        )

    def solve_static(
        self,
        hg: Optional[Array] = None,
        t: Array | float = 0.0,
        horseshoe: bool = False,
    ) -> AeroSnapshot:
        r"""
        Solve the VLM.
        :param hg: Beam global grid coordinates, [zeta_n, 4, 4]
        :param t: Time at which to solve static solution, used for flowfield evaluation
        :param horseshoe: If true, replace the wake with a horseshoe wake which extends a fixed distance
        # TODO: add free wake
        """

        case = self.initialise_case_object(1, horseshoe=horseshoe)
        case.t = case.t.at[0].set(t)

        out_case = self._solve(
            case, 0, hg, None, static=True, free_wake=False, horseshoe=horseshoe
        )[0]

        return out_case

    def solve_prescribed_dynamic(
        self,
        init_case: AeroSnapshot,
        hg_t: Array,
        hg_dot_t: Array,
        free_wake: bool = False,
    ) -> DynamicAeroCase:
        r"""
        Solve the UVLM for prescribed grid motions.
        :param init_case: StaticAero object containing initial conditions for the solution at time step 0.
        :param hg_t: Beam global grid coordinates over time, [n_tstep, zeta_n, 4, 4].
        :param hg_dot_t: Beam global grid velocities over time, [n_tstep, zeta_n, 4, 4].
        :param free_wake: If true, use free wake propagation.
        """
        check_arr_shape(hg_t, (None, None, 4, 4), "hg")
        check_if_all_se3_g(hg_t, True)
        if hg_dot_t is not None:
            if hg_t.shape != hg_dot_t.shape:
                raise ValueError(
                    f"hg_dot must have the same arr_list_shapes as hg, got {hg_dot_t.shape} vs {hg_t.shape}"
                )
        check_if_all_se3_a(hg_dot_t, True)

        n_tstep = hg_t.shape[0]

        case = init_case.to_dynamic(i_ts=0, n_tstep=n_tstep)

        def _step_func(i_ts_: int, case_: DynamicAeroCase) -> DynamicAeroCase:
            case_ = self._solve(
                case_,
                i_ts_,
                hg_t[i_ts_, ...],
                hg_dot_t[i_ts_, ...],
                static=False,
                free_wake=free_wake,
                horseshoe=False,
            )
            jax_print("UVLM timestep {i_ts_}", i_ts_=i_ts_)
            return case_

        case = fori_loop(
            1,
            n_tstep,
            _step_func,
            init_val=case,
        )
        return case

    def reference_snapshot(self) -> AeroSnapshot:
        r"""
        Get the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :return: StaticAero object at initial time step
        """
        return AeroSnapshot(
            zeta_b=self.zeta0_b,
            zeta_b_dot=ArrayList(
                [jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            zeta_w=self.zeta0_w,
            c=compute_c(self.zeta0_b),
            nc=compute_nc(self.zeta0_b),
            gamma_b=ArrayList([jnp.zeros((gd.m, gd.n)) for gd in self.grid_disc]),
            gamma_b_dot=ArrayList([jnp.zeros((gd.m, gd.n)) for gd in self.grid_disc]),
            gamma_w=ArrayList([jnp.zeros((gd.m_star, gd.n)) for gd in self.grid_disc]),
            f_steady=ArrayList(
                [jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            f_unsteady=ArrayList(
                [jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=-1,
            t=jnp.array((0.0,)),
            horseshoe=False,
            dof_mapping=self.dof_mapping,
            flowfield=self.flowfield,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            kernels=[*self.kernels_b, *self.kernels_w],
            aic_lu=jnp.zeros((sum(self.n_bound_panels), sum(self.n_bound_panels))),
            aic_piv=jnp.zeros((sum(self.n_bound_panels),)),
        )

    def plot_reference(
        self, directory: os.PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: Path to write files to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), plot_wake=plot_wake
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Get names of static attributes in UVLM
        :return: Sequence of static attribute names
        """
        return (
            "n_surf",
            "grid_disc",
            "n_bound_panels_tot",
            "n_wake_panels_tot",
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
        r"""
        Get names of dynamic attributes in UVLM
        :return: Sequence of dynamic attribute names
        """
        return (
            "_x0_b",
            "_zeta0_b",
            "_zeta0_w",
            "_dt",
            "_flowfield",
            "_delta_w",
        )
