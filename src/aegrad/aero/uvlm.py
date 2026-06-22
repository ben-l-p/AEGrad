from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from typing import Any, Callable, Optional, TYPE_CHECKING, Protocol
import os
from pathlib import Path
from functools import singledispatchmethod

import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import fori_loop

from aegrad.algebra.base import construct_approximation, jacrev_custom
from aegrad.structure.utils import transform_nodal_vect
from aegrad.utils.utils import make_pytree
from aegrad.algebra.test_routines import check_if_all_se3_g, check_if_all_se3_a
from aegrad.aero.utils import (
    propagate_wake,
    compute_c,
    compute_nc,
    calculate_steady_forcing,
    project_forcing_to_beam,
)
from aegrad.utils.constants import HORSESHOE_LENGTH
from aegrad.algebra.array_utils import (
    check_arr_dtype,
    neighbour_average,
    check_arr_shape,
    ArrayList,
    split_to_vertex,
)
from aegrad.aero.data_structures import (
    GridDiscretization,
    DynamicAeroCase,
    AeroSnapshot,
)
from aegrad.aero.flowfields import FlowField
from aegrad.aero.utils import KernelFunction, biot_savart_epsilon

from aegrad.algebra.se3 import vect_product as se3_vect_product
from aegrad.aero.aic import compute_v_ind, compute_aic_solve
from aegrad.aero.gradients.data_structures import (
    AeroDesignVariables,
    AeroJacobianApproximations,
    AeroStates,
    AeroGradsToCompute,
    GammaBApprox,
    GammaWApprox,
    ZetaWApprox,
    GammaBDotApprox,
    FAeroApprox,
)
from aegrad.utils.print_utils import VerbosityLevel
from aegrad.coupled.data_structures import AeroelasticDesignVariables
from aegrad.structure import BeamStructure, DynamicStructure
from aegrad.algebra.array_utils import construct_named_block_jacobian, ArrayListShape


if TYPE_CHECKING:
    from aegrad.aero.linear.linear_uvlm import LinearUVLM, LinearWakeType
from aegrad.utils.print_utils import warn, jax_print


class AeroGridFunction(Protocol):
    def __call__(self, x0: ArrayList, **kwargs: Array) -> ArrayList: ...


@make_pytree
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
        grid_func: Optional[AeroGridFunction] = None,
        free_wake: bool = False,
        gamma_dot_relaxation: float = 0.7,
        include_unsteady_force: bool = True,
        batch_size: int = 1,
    ) -> None:
        r"""
        Initialise UVLM class with all non-design parameters.
        :param grid_shapes: Discretisations for the number of chordwise, spanwise and wake-wise panels for each surface.
        May be passed as a sequence of either the GridDiscretization class or a tuple of integers ordered as (m, n,
        m_star).
        :param dof_mapping: Mapping from aerodynamic grid points to structure grid points for each surface.
        :param variable_wake_disc: If True, allow for variable wake discretisations.
        :param mirror_point: Optional point in mirror plane, [3]. If provided, this will apply mirroring of the aerodynamic
        geometry and flow about the plane defined by this point and the mirror normal.
        :param mirror_normal: Optional normal vector for mirror plane, [3].
        :param kernel: Input for custom kernel function to use for induced velocity calculations.
        :param grid_func: Input functions used for defining surfaces with control surfaces. This function should take
        the reference local grid coordinates ``x0_b``, as well as control inputs as keyword arguments, and return the
        deflected local grid coordinates as an ArrayList of equal dimensionality to ``x0_b``.
        :param free_wake: If True, include the velocity induced from the aerodynamic elements for wake propagation.
        :param gamma_dot_relaxation: Filtering parameter used for obtaining the time derivative of the circulation
        strengths.
        :param include_unsteady_force: If True, include forces due to apparent mass for simulation.
        :param batch_size: Batch size for vectorising AIC computations. Larger values may result in faster computations,
        at the expense of increased memory usage.
        """

        # case for single inputs
        if isinstance(dof_mapping, Array):
            dof_mapping_arraylist: ArrayList = ArrayList([dof_mapping])
        elif isinstance(dof_mapping, Sequence):
            dof_mapping_arraylist: ArrayList = ArrayList(dof_mapping)
        elif isinstance(dof_mapping, ArrayList):
            dof_mapping_arraylist: ArrayList = dof_mapping
        else:
            raise TypeError("Invalid dof mapping type")
        self.dof_mapping: ArrayList = ArrayList(dof_mapping_arraylist)

        # number of aerodynamic surfaces
        self.n_surf: int = len(grid_shapes)

        # set grid discretisations parameters for number of panels
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
        self._hg0: Optional[Array] = None
        self._x0_b: Optional[ArrayList] = None
        self._zeta0_b: Optional[ArrayList] = None
        self._zeta0_w: Optional[ArrayList] = None

        self.gamma_b_slice, self.gamma_w_slice = self._make_gamma_slices()

        # store DOF mapping
        if len(self.dof_mapping) != self.n_surf:
            raise ValueError(
                f"Expected {self.n_surf} DOF mapping arrays, got {len(self.dof_mapping)}"
            )
        for i_surf, map_ in enumerate(self.dof_mapping):
            check_arr_dtype(map_, int, "dof_mapping")
            check_arr_shape(map_, (self.grid_disc[i_surf].n + 1,), "grid_disc")

        # this must be optional as it is set as a design variable later
        self._flowfield: Optional[FlowField] = None

        # time step length
        self._dt: Optional[Array] = None

        # wake discretisation parameters
        self.variable_wake_disc: bool = variable_wake_disc
        self._delta_w: Optional[list[Optional[Array]]] = None

        # kernel definitions per surface (separate for wing and wake)
        self.kernels_b: Sequence[KernelFunction] = self.n_surf * [
            kernel if kernel is not None else biot_savart_epsilon
        ]
        self.kernels_w: Sequence[KernelFunction] = self.n_surf * [
            kernel if kernel is not None else biot_savart_epsilon
        ]

        # settings for solvers
        self.free_wake: bool = free_wake
        self.gamma_dot_relaxation: float = gamma_dot_relaxation
        self.include_unsteady_force: bool = include_unsteady_force
        self.batch_size: int = batch_size

        # mirror definitions
        if (mirror_point is None and mirror_normal is not None) or (
            mirror_point is not None and mirror_normal is None
        ):
            raise ValueError(
                "Both mirror_point and mirror_normal must be provided to apply mirroring, or both must be None to "
                "apply no mirroring."
            )

        if mirror_point is None or mirror_normal is None:
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

        def grid_func_inner(x0: ArrayList, **kwargs: Array) -> ArrayList:
            if grid_func is not None:
                return grid_func(x0, **kwargs)
            else:
                return x0

        # control surface variables
        self.grid_func: AeroGridFunction = grid_func_inner

        self.cs_ang0: dict[str, Array] = dict()
        self.cs_vel0: dict[str, Array] = dict()

    @property
    def flowfield(self) -> FlowField:
        r"""
        Get the FlowField object defining the background flow in space and time.
        :return: FlowField object.
        """
        if self._flowfield is None:
            raise ValueError("FlowField has not been set.")
        return self._flowfield

    @flowfield.setter
    def flowfield(self, flowfield: FlowField) -> None:
        self._flowfield = flowfield

    @property
    def x0_b(self) -> ArrayList:
        r"""
        Get the aerodynamic local grid coordinates at the reference configuration.
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3].
        """
        if self._x0_b is None:
            raise ValueError("Design variable x0_b has not been set.")
        return self._x0_b

    @property
    def zeta0_b(self) -> ArrayList:
        r"""
        Get the aerodynamic global grid coordinates at the reference configuration.
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3].
        """
        if self._zeta0_b is None:
            raise ValueError("Design variable zeta0_b has not been set.")
        return self._zeta0_b

    @property
    def zeta0_w(self) -> ArrayList:
        r"""
        Get the aerodynamic global wake grid coordinates at the reference configuration.
        :return: List of aerodynamic local grid coordinates for each surface, [n_surf][zeta_m_star, zeta_n, 3].
        """
        if self._zeta0_w is None:
            raise ValueError("Design variable zeta0_w has not been set.")
        return self._zeta0_w

    @property
    def dt(self) -> Array:
        r"""
        Get the time step length.
        :return: Time step length.
        """
        if self._dt is None:
            raise ValueError("Time step length dt has not been set.")
        return self._dt

    @property
    def delta_w(self) -> list[Optional[Array]]:
        r"""
        Get the wake displacement vector defining segment lengths of a variable wake discretisation per surface.
        :return: Wake displacement vectors for each surface, [n_surf][m_star].
        """
        if self._delta_w is None:
            raise ValueError("Wake displacement delta_w has not been set.")
        return self._delta_w

    @property
    def hg0(self) -> Array:
        r"""
        Get the reference beam coordinates used to create the reference aerodynamic grid.
        :return: Reference beam coordinates, [n_nodes, 4, 4].
        """
        if self._hg0 is None:
            raise ValueError("Variable hg0 has not been set.")
        return self._hg0

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
        :return: LinearUVLM model, linearised at specified time step.
        """

        # local import used to prevent circular import issues
        from aegrad.aero.linear.linear_uvlm import LinearUVLM, LinearWakeType

        return LinearUVLM(
            self,
            reference=reference,
            wake_type=wake_type if wake_type is not None else LinearWakeType.PRESCRIBED,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
            gamma_dot_state=gamma_dot_state,
        )

    def set_design_variables(
        self,
        dt: float | Array,
        flowfield: FlowField,
        x0_b: ArrayList | Sequence[Array] | Array,
        hg0: Array,
        delta_w: Optional[Sequence[Optional[Array]] | Optional[Array]] = None,
        reference_cs_angles: Optional[dict[str, Array]] = None,
    ) -> None:
        r"""
        Set aerodynamic design variables for solution.
        :param dt: Time step length
        :param flowfield: FlowField object defining the background flow in space and time
        :param delta_w: Vector to define segment lengths of a variable wake discretisation per surface. If None, this
        will use a uniform discretisation, as in the canonical UVLM.
        :param x0_b: Aerodynamic local grid coordinates, [n_surf][zeta_m, zeta_n, 3].
        :param hg0: Beam reference global grid coordinates, [n_nodes, 4, 4].
        dictionary input for deflections, and returns an ArrayList of the local deflected grid.
        :param reference_cs_angles: Dictionary of {name: angle} for each control surface at the reference. If None,
        defaults to no control surfaces.
        """

        if isinstance(delta_w, Array):
            delta_w_seq: Sequence[Optional[Array]] = [delta_w]
        elif delta_w is None:
            delta_w_seq = self.n_surf * [None]
        elif isinstance(delta_w, Sequence):
            delta_w_seq = delta_w
        else:
            raise TypeError("Invalid delta_w type")

        if isinstance(x0_b, Array):
            x0_aero_arraylist = ArrayList([x0_b])
        elif isinstance(x0_b, Sequence):
            x0_aero_arraylist = ArrayList(x0_b)
        elif isinstance(x0_b, ArrayList):
            x0_aero_arraylist = x0_b
        else:
            raise TypeError("Invalid x0_b type")

        # set aerodynamic local coordinates
        if len(x0_aero_arraylist) != self.n_surf:
            raise ValueError(
                f"Expected {self.n_surf} aerodynamic grid coordinate arrays, got {len(x0_b)}"
            )

        for i_surf in range(self.n_surf):
            check_arr_shape(
                x0_aero_arraylist[i_surf],
                (self.grid_disc[i_surf].m + 1, self.grid_disc[i_surf].n + 1, 3),
                "x0_b",
            )
        self._x0_b = x0_aero_arraylist

        if reference_cs_angles is not None:
            self.cs_ang0 = reference_cs_angles
            self.cs_vel0 = {
                k: jnp.zeros_like(v) for k, v in reference_cs_angles.items()
            }

        # set global grid coordinates for bound and wake
        check_arr_shape(hg0, (None, 4, 4), "hg0")
        self._hg0: Array = hg0
        self._zeta0_b = self.hg_to_zeta_b(hg_n=hg0, cs_ang_n=self.cs_ang0)

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
        for i_surf, dw_ in enumerate(delta_w_seq):
            if dw_ is None:
                self.delta_w.append(None)
            else:
                check_arr_shape(dw_, (self.grid_disc[i_surf].m_star,), "delta_w")
                self.delta_w.append(dw_)
        self._zeta0_w = self.initialise_wake()

    def case_from_dv(self, dv: AeroDesignVariables) -> UVLM:
        r"""
        Create a new UVLM instance as a function of design variables, allowing it to have defined gradients w.r.t.
        design variables.
        :param dv: Design variables.
        :return: UVLM object with the same functionality as ``self``.
        """
        inner_case = deepcopy(self)
        inner_case.set_design_variables(
            dt=self.dt,
            flowfield=inner_case.flowfield.from_design_variables(dv.flowfield)
            if dv.flowfield is not None
            else self.flowfield,
            delta_w=self.delta_w,
            x0_b=dv.x0_b if dv.x0_b is not None else self.x0_b,
            hg0=self.hg0,
            reference_cs_angles={k: v[0] for k, v in dv.cs_ang_t.items()}
            if dv.cs_ang_t is not None
            else self.cs_ang0,
        )

        return inner_case

    def get_design_variables(
        self,
        cs_ang_t: dict[str, Array],
        cs_vel_t: dict[str, Array],
        grads_to_compute: Optional[AeroGradsToCompute],
    ) -> AeroDesignVariables:
        r"""
        Extract design variables from the aerodynamic case. As the control input time histories are defined when
        initialising the simulation, they are not included in ``self`` and so are passed by argument.
        :param cs_ang_t: Time history of control surface angles, {keys, [n_tstep]}.
        :param cs_vel_t: Time history of control surface velocities, {keys, [n_tstep]}.
        :param grads_to_compute: Data structure which describes which design variables should be obtained. If None, all
        variables are obtained.
        :return: Aerodynamic design variables.
        """
        if isinstance(grads_to_compute, AeroGradsToCompute):
            return AeroDesignVariables(
                x0_b=self.x0_b if grads_to_compute.x0_aero else None,
                flowfield=self.flowfield.to_design_variables()
                if grads_to_compute.flowfield
                else None,
                cs_ang_t=cs_ang_t if grads_to_compute.cs_ang_t else None,
                cs_vel_t=cs_vel_t if grads_to_compute.cs_vel_t else None,
                f_shape=(),
            )
        else:  # grads_to_compute is None
            return AeroDesignVariables(
                x0_b=self.x0_b,
                flowfield=self.flowfield.to_design_variables(),
                cs_ang_t=cs_ang_t,
                cs_vel_t=cs_vel_t,
                f_shape=(),
            )

    def hg_to_zeta_b(self, hg_n: Array, cs_ang_n: dict[str, Array]) -> ArrayList:
        r"""
        Convert beam global grid coordinates to aerodynamic global grid coordinates.
        :param hg_n: Beam global grid coordinates at time step n, [n_nodes, 4, 4].
        :param cs_ang_n: Control surface angles as {surface_name: angle} pairs at time step n.
        :return: Full aerodynamic global grid coordinates for each surface, [n_surf][zeta_m, zeta_n, 3].
        """

        zeta_b0_cs = self.grid_func(
            self.x0_b, **cs_ang_n
        )  # get local aerodynamic grid for control surface deflections.

        zetas = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg = jnp.take(hg_n, self.dof_mapping[i_surf], axis=0)  # [varphi, 4, 4]

            zetas.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg, zeta_b0_cs[i_surf]
                )
            )
        return zetas

    def hg_dot_to_zeta_b_dot(
        self,
        hg_n: Array,
        hg_dot_n: Array,
        cs_ang_n: dict[str, Array],
        cs_vel_n: dict[str, Array],
    ) -> ArrayList:
        r"""
        Convert beam global grid velocities to aerodynamic global grid velocities.
        :param hg_n: Beam global grid coordinates, [n_nodes, 4, 4].
        :param hg_dot_n: Beam global grid velocities, [n_nodes, 4, 4].
        :param cs_ang_n: Control surface angles as {surface_name: angle} pairs.
        :param cs_vel_n: Control surface velocities as {surface_name: velocities} pairs.
        :return: Full aerodynamic global grid velocities for each surface, [n_surf][zeta_m, zeta_n, 3].
        """
        zeta_b0_cs = self.grid_func(
            self.x0_b, **cs_ang_n
        )  # deflected local aerodynamic grid

        # as this is where the control velocities are used, they are checked here
        for key in set(cs_ang_n.keys()) | set(cs_vel_n.keys()):
            if key not in cs_vel_n or key not in cs_ang_n:
                raise ValueError(
                    f"Missing pair of control angles and velocities for control surface key '{key}'"
                )

            if cs_vel_n[key].shape != cs_vel_n[key].shape:
                raise ValueError(
                    f"Mismatched shape for control surface angles {cs_vel_n[key].shape} and velocities {cs_vel_n[key].shape}"
                )

        # use jvp to find the velocity of the aerodynamic grid due to control velocity.
        _, zeta_b0_dot_cs = jax.jvp(
            lambda angs: self.grid_func(
                self.x0_b, **angs
            ),  # local grid velocities due to control surface
            primals=(cs_ang_n,),
            tangents=(cs_vel_n,),
        )

        zeta_dots = ArrayList([])
        for i_surf in range(self.n_surf):
            this_hg = jnp.take(hg_n, self.dof_mapping[i_surf], axis=0)  # [varphi, 4, 4]
            this_hg_dot = jnp.take(
                hg_dot_n, self.dof_mapping[i_surf], axis=0
            )  # [varphi, 4, 4]
            this_rmat = this_hg[:, :3, :3]  # [n_span, 3, 3]
            zeta_dots.append(
                vmap(vmap(se3_vect_product, (None, 0), 0), (0, 1), 1)(
                    this_hg_dot, zeta_b0_cs[i_surf]
                )
                + jnp.einsum(
                    "njk,mnk->mnj",
                    this_rmat,
                    zeta_b0_dot_cs[i_surf],
                )
            )
        return zeta_dots

    def _make_gamma_slices(self) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        r"""
        Create slices for the bound and wake circulation strengths in their respective solution vectors based on the
        grid discretisations.
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
        self, zeta_b: Array, i_surf: int, horseshoe_length: float
    ) -> Array:
        r"""
        Create a horseshoe wake grid for a given surface at a given time step.
        :param zeta_b: Bound grid coordinates for single surface, [zeta_m, zeta_n, 3].
        :param i_surf: Surface index.
        :param horseshoe_length: Length of horseshoe wake to generate.
        :return: Wake grid coordinates, [2, zeta_n, 3]
        """
        zeta_te = zeta_b[-1, ...]  # [zeta_n, 3]
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
        zeta_b: ArrayList = zeta_b if zeta_b is not None else self.zeta0_b

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

    @staticmethod
    def compute_gamma_dot(
        gamma_b_n: ArrayList,
        gamma_b_nm1: ArrayList,
        gamma_b_dot_nm1: ArrayList,
        dt: Array,
        gamma_dot_relaxation: float,
    ) -> ArrayList:
        r"""
        Calculate time derivative of bound circulation strengths at specified time step using finite difference.
        :param gamma_b_n: Bound circulation strengths at timestep n, [n_surf][m, n].
        :param gamma_b_nm1: Bound circulation strengths at timestep n-1, [n_surf][m, n].
        :param gamma_b_dot_nm1: Filtered bound circulation strengths time derivative at timestep n-1, [n_surf][m, n].
        :param dt: Time step length.
        :param gamma_dot_relaxation: Relaxation factor which filters the time derivative.
        """

        # first obtain the current unfiltered, and previous filtered values for gamma_dot
        gamma_b_dot_curr = (gamma_b_n - gamma_b_nm1) / dt

        # blend with relaxation parameter
        return (
            gamma_dot_relaxation * gamma_b_dot_curr
            + (1.0 - gamma_dot_relaxation) * gamma_b_dot_nm1
        )

    @singledispatchmethod
    def set_gamma_w(self, gamma_vec: Array, case: DynamicAeroCase, i_ts: int) -> None:
        r"""
        Set wake circulation strengths from total circulation strengths at specified time step. Can be passed either a
        full vector of strengths, or a sequence of strengths per surface.
        :param case: DynamicAeroCase object.
        :param gamma_vec: Total circulation strengths vector, [gamma_w_tot].
        :param i_ts: Timestep index.
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

    def base_solve(
        self,
        q_nm1: Optional[AeroStates],
        t_n: Array,
        hg_n: Optional[Array],
        hg_nm1: Optional[Array],
        hg_dot_n: Optional[Array],
        static: bool,
        horseshoe: bool,
        cs_ang_n: dict[str, Array],
        cs_ang_nm1: Optional[dict[str, Array]],
        cs_vel_n: Optional[dict[str, Array]],
    ) -> tuple[
        ArrayList,
        ArrayList,
        ArrayList,
        ArrayList,
        Optional[ArrayList],
        ArrayList,
        ArrayList,
        Optional[ArrayList],
        ArrayList,
        Optional[ArrayList],
    ]:
        r"""
        Solve the UVLM equations for a single time step. Can be used for both static and dynamic solves.
        :param q_nm1: Minimal aerodynamic states from timestep n-1.
        :param t_n: Time at timestep n.
        :param hg_n: Beam global grid coordinates at time step n, [n_nodes, 4, 4].
        :param hg_nm1: Beam global grid coordinates at time step n - 1, [n_nodes, 4, 4]. Required for consistent free
        wake modelling.
        :param hg_dot_n: Beam global grid velocities, [n_nodes, 4, 4].
        :param static: If True, perform a static solve.
        :param horseshoe: If True, replace the wake with a horseshoe wake in static solve which extends a fixed
        distance.
        :param cs_ang_n: Control surface angle at timestep n, {name, []}.
        :param cs_ang_nm1: Control surface angle at timestep n - 1, {name, []}.
        :param cs_vel_n: Control surface velocity at timestep n, {name, []}.
        :return: Collocation points, bound normals, bound circulation, wake circulation, bound circulation time
        derivative, bound grid, wake grid, bound grid time derivattive, steady forcing and unsteady forcing.
        """
        if not (0.0 < self.gamma_dot_relaxation <= 1.0):
            warn("Gamma_dot relaxation factor not in (0, 1]")

        if not static and horseshoe:
            warn(
                "Horseshoe wake not compatible with non-static solve. Overriding horseshoe to False."
            )
            horseshoe = False

        if not static and q_nm1 is None:
            raise ValueError("q_nm1 needs to be specified for dynamic solve")

        zeta_b_n = self.hg_to_zeta_b(
            hg_n=hg_n if hg_n is not None else self.hg0, cs_ang_n=cs_ang_n
        )

        c_n = compute_c(zetas=zeta_b_n)
        nc_n = compute_nc(zetas=zeta_b_n)

        if hg_dot_n is None:
            zeta_b_dot_n: Optional[ArrayList] = None
            c_dot_n: Optional[ArrayList] = None
            zeta_b_nm1: Optional[ArrayList] = None
        else:
            assert (
                hg_n is not None
                and cs_vel_n is not None
                and hg_nm1 is not None
                and cs_ang_nm1 is not None
            )
            zeta_b_dot_n = self.hg_dot_to_zeta_b_dot(
                hg_n=hg_n, hg_dot_n=hg_dot_n, cs_ang_n=cs_ang_n, cs_vel_n=cs_vel_n
            )

            zeta_b_nm1 = self.hg_to_zeta_b(hg_n=hg_nm1, cs_ang_n=cs_ang_nm1)

            c_dot_n = ArrayList(
                [neighbour_average(zeta_dot, axes=(0, 1)) for zeta_dot in zeta_b_dot_n]
            )

        if static:
            # initialise wake
            # update zeta0_w
            if horseshoe:
                zeta_w_n = ArrayList(
                    [
                        self._make_surf_horseshoe_wake(
                            zeta_b=zeta_b_n[i_surf],
                            i_surf=i_surf,
                            horseshoe_length=HORSESHOE_LENGTH,
                        )
                        for i_surf in range(self.n_surf)
                    ]
                )
            else:
                # Re-initialise wake. This is wasteful when there is no coupled structure as zeta_ws will equal zeta0_w.
                # It is necessary to update the wake grid coordinates when there is a coupled structure as the bound
                # grid coordinates will have changed from the initial configuration.
                zeta_w_n = self.initialise_wake(zeta_b_n)

            gamma_w_n = None  # allocate later from gamma_b
        else:
            assert q_nm1 is not None and zeta_b_nm1 is not None

            zeta_full = ArrayList([*zeta_b_nm1, *q_nm1.zeta_w])
            gamma_full = ArrayList([*q_nm1.gamma_b, *q_nm1.gamma_w])

            def v_wake_prop(x_: Array) -> Array:
                v = self.flowfield.vmap_call(x=x_, t=t_n)
                if self.free_wake:
                    v += compute_v_ind(
                        cs=x_,
                        zetas=zeta_full,
                        gammas=gamma_full,
                        kernels=[*self.kernels_b, *self.kernels_w],
                        batch_size=self.batch_size,
                        mirror_normal=self.mirror_normal,
                        mirror_point=self.mirror_point,
                    )
                return v

            # propagate wake
            zeta_w_n, gamma_w_n = propagate_wake(
                gamma_b_nm1=q_nm1.gamma_b,
                gamma_w_nm1=q_nm1.gamma_w,
                zeta_b_n=zeta_b_n,
                zeta_w_nm1=q_nm1.zeta_w,
                delta_w=self.delta_w,
                v_func=v_wake_prop,
                dt=self.dt,
                frozen_wake=False,
                linearise_variable_wake=False,
            )

        aic_solve = compute_aic_solve(
            cs=c_n,
            ns=nc_n,
            zetas_b=zeta_b_n,
            zetas_w=zeta_w_n if static else None,
            kernels_b=self.kernels_b,
            kernels_w=self.kernels_w if static else None,
            batch_size=self.batch_size,
            mirror_normal=self.mirror_normal,
            mirror_point=self.mirror_point,
        )

        v_bc_n = self.flowfield.surf_vmap_call(xs=c_n, t=t_n)  # [n_surf][m, varphi, 3]

        if not static:
            # structural component
            v_bc_n -= c_dot_n

            if zeta_w_n is None or gamma_w_n is None:
                raise ValueError("zeta_w_nm1 and gamma_w_nm1 are None")

            # find wake component
            v_bc_n += compute_v_ind(
                cs=c_n,
                zetas=zeta_w_n,
                gammas=gamma_w_n,
                kernels=self.kernels_w,
                batch_size=self.batch_size,
                mirror_normal=self.mirror_normal,
                mirror_point=self.mirror_point,
            )

        v_bc_n = ArrayList.einsum("ijk,ijk->ij", v_bc_n, nc_n)  # [c_tot]

        gamma_b_vec_n = jnp.linalg.solve(aic_solve, -v_bc_n.ravel())

        # assemble back to surface ArrayList
        gamma_b_n = ArrayList([])
        for i_surf in range(self.n_surf):
            gamma_b_n.append(
                gamma_b_vec_n[self.gamma_b_slice[i_surf]].reshape(
                    self.grid_disc[i_surf].m, self.grid_disc[i_surf].n
                )
            )

        if static:
            gamma_b_dot_n: Optional[ArrayList] = None
            f_unsteady: Optional[ArrayList] = None
        else:
            if q_nm1 is None:
                raise ValueError("q_nm1 needs to be specified for dynamic solve")

            gamma_b_dot_n = self.compute_gamma_dot(
                gamma_b_n=gamma_b_n,
                gamma_b_nm1=q_nm1.gamma_b,
                gamma_b_dot_nm1=q_nm1.gamma_b_dot,
                dt=self.dt,
                gamma_dot_relaxation=self.gamma_dot_relaxation,
            )
            f_unsteady: Optional[ArrayList] = ArrayList(
                [
                    split_to_vertex(
                        self.flowfield.rho
                        * gamma_b_dot_n[i_surf][..., None]
                        * nc_n[i_surf],
                        (0, 1),
                    )
                    for i_surf in range(self.n_surf)
                ]
            )

        if static:
            # wake circulation is the same as trailing edge bound circulation
            gamma_w_n: ArrayList = ArrayList(
                [
                    jnp.broadcast_to(
                        gb[[-1], ...], shape=(1 if horseshoe else gd.m_star, gd.n)
                    )
                    for gb, gd in zip(gamma_b_n, self.grid_disc)
                ]
            )

        if gamma_w_n is None:
            raise ValueError("gamma_w_nm1 is None")
        if zeta_w_n is None:
            raise ValueError("zeta_w_nm1 is None")

        # Steady forces: total velocity (background + all-surface induced) minus grid velocity.
        # Use zeros for grid velocity in the static case (fixed grid).
        zeta_b_dot_for_forces = (
            zeta_b_dot_n
            if zeta_b_dot_n is not None
            else ArrayList([jnp.zeros_like(zb) for zb in zeta_b_n])
        )

        def v_total_func(x_: Array) -> Array:
            return self.flowfield.vmap_call(x=x_, t=t_n) + compute_v_ind(
                cs=x_,
                zetas=ArrayList([*zeta_b_n, *zeta_w_n]),
                gammas=ArrayList([*gamma_b_n, *gamma_w_n]),
                kernels=[*self.kernels_b, *self.kernels_w],
                batch_size=self.batch_size,
                mirror_normal=self.mirror_normal,
                mirror_point=self.mirror_point,
            )

        f_steady = calculate_steady_forcing(
            zeta_b=zeta_b_n,
            zeta_dot_b=zeta_b_dot_for_forces,
            gamma_b=gamma_b_n,
            gamma_w=gamma_w_n,
            rho=self.flowfield.rho,
            v_func=v_total_func,
            v_inputs=None,
        )

        return (
            c_n,
            nc_n,
            gamma_b_n,
            gamma_w_n,
            gamma_b_dot_n,
            zeta_b_n,
            zeta_w_n,
            zeta_b_dot_n,
            f_steady,
            f_unsteady,
        )

    def case_solve(
        self,
        case: DynamicAeroCase,
        i_ts: int,
        hg_n: Optional[Array],
        hg_nm1: Optional[Array],
        hg_dot_n: Optional[Array],
        static: bool,
        horseshoe: bool,
        cs_ang_n: dict[str, Array],
        cs_ang_nm1: Optional[dict[str, Array]],
        cs_vel_n: Optional[dict[str, Array]],
    ) -> DynamicAeroCase:
        r"""
        Solve the UVLM equations for a single time step. Can be used for both static and dynamic solves. The solution
        is updated in-place in the case object.
        :param case: Solution object.
        :param i_ts: Timestep index to solve for.
        :param hg_n: Beam global grid coordinates at time step n, [zeta_n, 4, 4].
        :param hg_nm1: Beam global grid coordinates at time step n11, [zeta_n, 4, 4].
        :param hg_dot_n: Beam global grid velocities at time step n, [zeta_n, 4, 4].
        :param static: If true, perform a static solve.
        :param horseshoe: If true, replace the wake with a static_horseshoe wake in static solve which extends a fixed
        distance.
        :param cs_ang_n: Control surface angle at timestep n, {name, []}.
        :param cs_ang_nm1: Control surface angle at timestep n - 1, {name, []}.
        :param cs_vel_n: Control surface velocity at timestep n, {name, []}.
        :return: Solution object with data for current time step added.
        """

        q_nm1 = AeroStates(
            gamma_b=case.gamma_b.index_all(i_ts - 1, ...),
            gamma_w=case.gamma_w.index_all(i_ts - 1, ...),
            gamma_b_dot=case.gamma_b_dot.index_all(i_ts - 1, ...),
            zeta_w=case.zeta_w.index_all(i_ts - 1, ...),
        )

        if not static:
            case.t = case.t.at[i_ts].set(
                jax.lax.select(i_ts, case.t[i_ts - 1] + self.dt, 0.0)
            )

        (
            c_n,
            nc_n,
            gamma_b_n,
            gamma_w_n,
            gamma_b_dot_n,
            zeta_b_n,
            zeta_w_n,
            zeta_b_dot_n,
            f_steady,
            f_unsteady,
        ) = self.base_solve(
            q_nm1=q_nm1,
            t_n=case.t[i_ts, ...],
            hg_n=hg_n,
            hg_nm1=hg_nm1,
            hg_dot_n=hg_dot_n,
            static=static,
            horseshoe=horseshoe,
            cs_ang_n=cs_ang_n,
            cs_ang_nm1=cs_ang_nm1,
            cs_vel_n=cs_vel_n,
        )

        case.set_arraylist_at_ts("_c", values=c_n, i_ts=i_ts)
        case.set_arraylist_at_ts("_nc", values=nc_n, i_ts=i_ts)
        case.set_arraylist_at_ts("_gamma_b", values=gamma_b_n, i_ts=i_ts)
        case.set_arraylist_at_ts("_gamma_w", values=gamma_w_n, i_ts=i_ts)
        case.set_arraylist_at_ts("_zeta_b", values=zeta_b_n, i_ts=i_ts)
        case.set_arraylist_at_ts("_f_steady", values=f_steady, i_ts=i_ts)

        if not static:
            if gamma_b_dot_n is None:
                raise ValueError("gamma_b_dot_n is None")
            if zeta_b_dot_n is None:
                raise ValueError("zeta_b_dot_n is None")
            if f_unsteady is None:
                raise ValueError("f_unsteady is None")
            case.set_arraylist_at_ts("_gamma_b_dot", values=gamma_b_dot_n, i_ts=i_ts)
            case.set_arraylist_at_ts("_zeta_b_dot", values=zeta_b_dot_n, i_ts=i_ts)
            case.set_arraylist_at_ts("_f_unsteady", values=f_unsteady, i_ts=i_ts)

        # set wake grid coordinates. If using static_horseshoe, it will still create a regular wake for plotting
        if horseshoe:
            case.set_arraylist_at_ts(
                "_zeta_w", values=self.initialise_wake(zeta_w_n), i_ts=i_ts
            )
        else:
            case.set_arraylist_at_ts("_zeta_w", values=zeta_w_n, i_ts=i_ts)

        return case

    def initialise_case_object(
        self,
        n_tstep: int,
        static_horseshoe: bool,
        free_wake: bool,
        gamma_dot_relaxation: float,
        cs_ang_t: dict[str, Array],
        cs_vel_t: dict[str, Array],
    ) -> DynamicAeroCase:
        r"""
        Initialise an DynamicAeroCase object to store the solution of the aerodynamic case for a given number of time
        steps. All solution data is initialised to zero.
        :param n_tstep: Number of time steps to solve for in dynamic solution.
        :param static_horseshoe: Whether a horseshoe formulation was used for the static case.
        :param free_wake: Whether to use a free wake formulation.
        :param gamma_dot_relaxation: Relaxation factor for damping gamma_dot.
        :param cs_ang_t: Control surface angle time history, {name, [n_tstep]}.
        :param cs_vel_t: Control surface velocity time history, {name, [n_tstep]}.
        :return: DynamicAeroCase object.
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
            n=ArrayList([jnp.zeros((n_tstep, gd.m, gd.n, 3)) for gd in self.grid_disc]),
            kernels=[*self.kernels_b, *self.kernels_w],
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=jnp.arange(n_tstep),
            t=jnp.zeros(n_tstep),
            dof_mapping=self.dof_mapping,
            static_horseshoe=static_horseshoe,
            free_wake=free_wake,
            gamma_dot_relaxation=gamma_dot_relaxation,
            cs_ang=cs_ang_t,
            cs_vel=cs_vel_t,
            batch_size=self.batch_size,
        )

    def solve_static(
        self,
        hg: Optional[Array] = None,
        t: Array | float = 0.0,
        horseshoe: bool = False,
        cs_ang: Optional[dict[str, Array]] = None,
    ) -> AeroSnapshot:
        r"""
        Solve the VLM for given static beam coordinates.
        TODO: add free wake
        :param hg: Beam coordinates, [zeta_n, 4, 4].
        :param t: Time at which to solve static solution, used for background flowfield evaluation.
        :param horseshoe: If true, replace the wake with a horseshoe wake which extends a fixed distance.
        :param cs_ang: Control surface angles, {name, []}.
        :return: AeroSnapshot solution object.
        """

        case = self.initialise_case_object(
            1,
            static_horseshoe=horseshoe,
            gamma_dot_relaxation=0.7,
            free_wake=False,
            cs_ang_t=cs_ang if cs_ang is not None else self.cs_ang0,
            cs_vel_t={
                k: jnp.zeros_like(v)
                for k, v in (cs_ang if cs_ang is not None else self.cs_ang0).items()
            },
        )
        case.t = case.t.at[0].set(t)

        out_case = self.case_solve(
            case=case,
            i_ts=0,
            hg_n=hg,
            hg_nm1=None,
            hg_dot_n=None,
            static=True,
            horseshoe=horseshoe,
            cs_ang_n=cs_ang if cs_ang is not None else self.cs_ang0,
            cs_ang_nm1=None,
            cs_vel_n=None,
        )[0]

        if horseshoe:
            # if using a horseshoe wake, return the normal wake to prevent continuity errors
            out_case.zeta_w = self.initialise_wake(zeta_b=out_case.zeta_b)

        return out_case

    def solve_prescribed_dynamic(
        self,
        init_case: AeroSnapshot,
        hg_t: Array,
        hg_dot_t: Array,
        cs_ang_t: Optional[dict[str, Array]] = None,
        cs_vel_t: Optional[dict[str, Array]] = None,
    ) -> DynamicAeroCase:
        r"""
        Solve the UVLM for prescribed grid motions.
        :param init_case: StaticAero object containing initial conditions for the solution at time step 0.
        :param hg_t: Beam coordinates over time, [n_tstep, n_nodes, 4, 4].
        :param hg_dot_t: Beam coordinate time derivative over time, [n_tstep, n_nodes, 4, 4].
        :param cs_ang_t: Control surface angle time history, {name, [n_tstep]}.
        :param cs_vel_t: Control surface velocity time history, {name, [n_tstep]}.
        :return: DynamicAeroCase solution object.
        """
        check_arr_shape(hg_t, (None, None, 4, 4), "hg_n")
        check_if_all_se3_g(hg_t, True)

        if hg_t.shape != hg_dot_t.shape:
            raise ValueError(
                f"hg_dot_n must have the same arr_list_shapes as hg_n, got {hg_dot_t.shape} vs {hg_t.shape}"
            )

        check_if_all_se3_a(hg_dot_t, True)

        n_tstep = hg_t.shape[0]

        case = init_case.to_dynamic(i_ts=0, n_tstep=n_tstep)

        def _step_func(i_ts_: int, case_: DynamicAeroCase) -> DynamicAeroCase:
            cs_angle_nm1, cs_angle_n = (
                (
                    {k: v[i_ts__] for k, v in cs_ang_t.items()}
                    if cs_ang_t is not None
                    else dict()
                )
                for i_ts__ in (i_ts_ - 1, i_ts_)
            )
            cs_velocity_n = (
                {k: v[i_ts_] for k, v in cs_vel_t.items()}
                if cs_vel_t is not None
                else dict()
            )

            case_ = self.case_solve(
                case=case_,
                i_ts=i_ts_,
                hg_n=hg_t[i_ts_, ...],
                hg_nm1=hg_t[i_ts_ - 1, ...],
                hg_dot_n=hg_dot_t[i_ts_, ...],
                static=False,
                horseshoe=False,
                cs_ang_n=cs_angle_n,
                cs_ang_nm1=cs_angle_nm1,
                cs_vel_n=cs_velocity_n,
            )
            jax_print(
                "UVLM timestep {i_ts_}",
                i_ts_=i_ts_,
                verbose_level=VerbosityLevel.NORMAL,
            )
            return case_

        case = fori_loop(
            1,
            n_tstep,
            _step_func,
            init_val=case,
        )
        return case

    def reference_configuration(self) -> AeroSnapshot:
        r"""
        Get the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :return: StaticAero object at initial time step.
        """
        return AeroSnapshot(
            zeta_b=self.zeta0_b,
            zeta_b_dot=ArrayList(
                [jnp.zeros((gd.m + 1, gd.n + 1, 3)) for gd in self.grid_disc]
            ),
            zeta_w=self.zeta0_w,
            c=compute_c(self.zeta0_b),
            n=compute_nc(self.zeta0_b),
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
            dof_mapping=self.dof_mapping,
            flowfield=self.flowfield,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            kernels=[*self.kernels_b, *self.kernels_w],
            static_horseshoe=False,
            gamma_dot_relaxation=0.0,
            free_wake=False,
            cs_ang=self.cs_ang0,
            cs_vel=self.cs_vel0,
            batch_size=self.batch_size,
        )

    def plot_reference(
        self, directory: os.PathLike | str, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: Path to write files to.
        :param plot_wake: If True, plot the wake grid.
        """
        return self.reference_configuration().plot(
            Path(directory).resolve(), plot_wake=plot_wake
        )

    def _base_solve_from_dv(
        self,
        hg_n: Array,
        hg_nm1: Array,
        hg_dot_n: Array,
        t_n: Array,
        q_nm1: AeroStates,
        dv: AeroDesignVariables,
        cs_ang_n: dict[str, Array],
        cs_ang_nm1: Optional[dict[str, Array]],
        cs_vel_n: Optional[dict[str, Array]],
    ) -> tuple[ArrayList, ArrayList, ArrayList, ArrayList, Array]:
        r"""
        Perform base_solve from design variables and return intermediate quantities.
        :param hg_n: Beam coordinates at time step n, [n_nodes, 4, 4].
        :param hg_nm1: Beam coordinates at time step n-1, [n_nodes, 4, 4].
        :param hg_dot_n: Beam velocities at time step n, [n_nodes, 4, 4].
        :param t_n: Time at step n.
        :param q_nm1: Aero minimal states at timestep n-1.
        :param dv: Aero design variables.
        :param cs_ang_n: Control surface angles at time step n, {name, []}.
        :param cs_ang_nm1: Control surface angles at time step n-1, {name, []}.
        :param cs_vel_n: Control velocity at time step n-1, {name, []}.
        :return: Bound circulation, wake circulation, bound circulation time derivative, wake grid and aerodynamic
        force projected onto local beam coordinates.
        """
        inner_case = self.case_from_dv(dv=dv)

        (
            c_n,
            nc_n,
            gamma_b_n,
            gamma_w_n,
            gamma_b_dot_n,
            zeta_b_n,
            zeta_w_n,
            zeta_b_dot_n,
            f_steady,
            f_unsteady,
        ) = inner_case.base_solve(
            q_nm1=q_nm1,
            t_n=t_n,
            hg_n=hg_n,
            hg_nm1=hg_nm1,
            hg_dot_n=hg_dot_n,
            static=False,
            horseshoe=False,
            cs_ang_n=cs_ang_n,
            cs_ang_nm1=cs_ang_nm1,
            cs_vel_n=cs_vel_n,
        )

        assert gamma_b_dot_n is not None and f_unsteady is not None

        # project forcing onto beam
        f_total_zeta = f_steady + f_unsteady
        f_aero_beam_global = project_forcing_to_beam(
            f_total=f_total_zeta,
            rmat=hg_n[:, :3, :3],
            dof_mapping=self.dof_mapping,
            x0_aero=inner_case.x0_b,
        )
        f_aero_beam_local = transform_nodal_vect(
            vect=f_aero_beam_global, rmat=jnp.swapaxes(hg_n[:, :3, :3], -2, -1)
        )

        return gamma_b_n, gamma_w_n, gamma_b_dot_n, zeta_w_n, f_aero_beam_local

    def gamma_b_res_func(
        self,
        i_ts: int,
        t_n: Array,
        varphi_n: Array,
        v_n: Array,
        gamma_b_n: Array,
        gamma_w_n: Array,
        zeta_w_n: Array,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        struct_obj: BeamStructure,
    ) -> Array:
        r"""
        Bound circulation residual used for adjoint computations.

        :math:`\mathbf{r}_{\Gamma_b} = \left(\boldsymbol{\mathcal{A}}_{b, n} \cdot \mathbf{n}_n\right)^{-1} \left[\left(
        \boldsymbol{\mathcal{A}}_{w, n} \boldsymbol{\Gamma}_{w, n} +\mathbf{v}_{bc, n} - \dot{\boldsymbol{\zeta}}_c\right)
        \cdot \mathbf{n}_n\right] + \boldsymbol{\Gamma}_{b, n}`

        :param i_ts: Time step index.
        :param t_n: Time at step n.
        :param varphi_n: varphi vector at timestep n.
        :param v_n: Beam velocity vector at timestep n.
        :param gamma_b_n: Bound circulation vector at timestep n.
        :param gamma_w_n: Wake circulation vector at timestep n.
        :param zeta_w_n: Wake grid vector at timestep n.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions for the variables where gradients aren't
        requested.
        :param struct_obj: Beam structure.
        :return: Bound circulation residual.
        """

        varphi_n = varphi_n.reshape(-1, 6)
        v_n = v_n.reshape(-1, 6)
        gamma_b_n = ArrayList.from_vector(
            vect=gamma_b_n,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )
        gamma_w_n = ArrayList.from_vector(
            vect=gamma_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star, gd.n) for gd in self.grid_disc]
            ),
        )
        zeta_w_n = ArrayList.from_vector(
            vect=zeta_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star + 1, gd.n + 1, 3) for gd in self.grid_disc]
            ),
        )

        inner_struct = struct_obj.case_from_dv(dv=dv.structure)
        hg_n = inner_struct.calculate_hg_from_varphi(varphi=varphi_n)
        hg_dot_n = inner_struct.make_hg_dot(hg=hg_n, v=v_n)

        inner_case = self.case_from_dv(dv=dv.aero)

        # get control surface deflections from design variables
        cs_ang_n, cs_vel_n = dv.aero.get_cs_n(i_ts=i_ts, dv_full=dv_full.aero)

        zeta_b_n = inner_case.hg_to_zeta_b(hg_n=hg_n, cs_ang_n=cs_ang_n)

        c_n = compute_c(zetas=zeta_b_n)
        nc_n = compute_nc(zetas=zeta_b_n)

        zeta_b_dot_n = inner_case.hg_dot_to_zeta_b_dot(
            hg_n=hg_n, hg_dot_n=hg_dot_n, cs_ang_n=cs_ang_n, cs_vel_n=cs_vel_n
        )

        c_dot_n = ArrayList(
            [neighbour_average(zeta_dot, axes=(0, 1)) for zeta_dot in zeta_b_dot_n]
        )

        aic_solve = compute_aic_solve(
            cs=c_n,
            ns=nc_n,
            zetas_b=zeta_b_n,
            zetas_w=None,
            kernels_b=inner_case.kernels_b,
            kernels_w=None,
            batch_size=self.batch_size,
            mirror_normal=inner_case.mirror_normal,
            mirror_point=inner_case.mirror_point,
        )

        v_bc_n = inner_case.flowfield.surf_vmap_call(
            xs=c_n, t=t_n
        )  # [n_surf][m, varphi, 3]

        # structural component
        v_bc_n -= c_dot_n

        # find wake component
        v_bc_n += compute_v_ind(
            cs=c_n,
            zetas=zeta_w_n,
            gammas=gamma_w_n,
            kernels=inner_case.kernels_w,
            batch_size=self.batch_size,
            mirror_normal=inner_case.mirror_normal,
            mirror_point=inner_case.mirror_point,
        )

        v_bc_n = ArrayList.einsum("ijk,ijk->ij", v_bc_n, nc_n)  # [c_tot]

        gamma_b_vec_n = jnp.linalg.solve(aic_solve, -v_bc_n.ravel())

        # assemble back to surface ArrayList
        gamma_b_nm1_update = ArrayList([])
        for i_surf in range(inner_case.n_surf):
            gamma_b_nm1_update.append(
                gamma_b_vec_n[inner_case.gamma_b_slice[i_surf]].reshape(
                    inner_case.grid_disc[i_surf].m, inner_case.grid_disc[i_surf].n
                )
            )

        return (gamma_b_nm1_update - gamma_b_n).ravel()

    def wake_prop_res_func(
        self,
        i_ts: int,
        t_n: Array,
        varphi_nm1: Array,
        varphi_n: Array,
        gamma_b_nm1: Array,
        gamma_w_nm1: Array,
        gamma_w_n: Array,
        zeta_w_nm1: Array,
        zeta_w_n: Array,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        struct_obj: BeamStructure,
    ) -> tuple[Array, Array]:
        r"""
        Wake propagation residual for both grid coordinates and circulation strengths.

        :param i_ts: Time step index.
        :param t_n: Time at timestep n.
        :param varphi_nm1: Beam minimal coordinates vector at timestep n-1.
        :param varphi_n: Beam maximal coordinates vector at timestep n.
        :param gamma_b_nm1: Bound circulation vector at timestep n-1.
        :param gamma_w_nm1: Wake circulation vector at timestep n-1.
        :param gamma_w_n: Wake circulation vector at timestep n.
        :param zeta_w_nm1: Wake grid vector at timestep n-1.
        :param zeta_w_n: Wake grid vector at timestep n.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions.
        :param struct_obj: Beam structure.
        :return: Wake grid and circulation residuals.
        """

        varphi_nm1 = varphi_nm1.reshape(-1, 6)
        varphi_n = varphi_n.reshape(-1, 6)

        gamma_b_nm1 = ArrayList.from_vector(
            vect=gamma_b_nm1,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        gamma_w_nm1 = ArrayList.from_vector(
            vect=gamma_w_nm1,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star, gd.n) for gd in self.grid_disc]
            ),
        )

        gamma_w_n = ArrayList.from_vector(
            vect=gamma_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star, gd.n) for gd in self.grid_disc]
            ),
        )
        zeta_w_nm1 = ArrayList.from_vector(
            vect=zeta_w_nm1,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star + 1, gd.n + 1, 3) for gd in self.grid_disc]
            ),
        )

        zeta_w_n = ArrayList.from_vector(
            vect=zeta_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star + 1, gd.n + 1, 3) for gd in self.grid_disc]
            ),
        )

        inner_struct = struct_obj.case_from_dv(dv=dv.structure)
        hg_nm1 = inner_struct.calculate_hg_from_varphi(varphi=varphi_nm1)
        hg_n = inner_struct.calculate_hg_from_varphi(varphi=varphi_n)

        inner_case = self.case_from_dv(dv=dv.aero)

        # get control surface deflections from design variables
        cs_ang_nm1, cs_vel_nm1 = dv.aero.get_cs_n(i_ts=i_ts - 1, dv_full=dv_full.aero)
        cs_ang_n, cs_vel_n = dv.aero.get_cs_n(i_ts=i_ts, dv_full=dv_full.aero)
        zeta_b_nm1 = inner_case.hg_to_zeta_b(hg_n=hg_nm1, cs_ang_n=cs_ang_nm1)
        zeta_b_n = inner_case.hg_to_zeta_b(hg_n=hg_n, cs_ang_n=cs_ang_n)

        def v_wake_prop(x_: Array) -> Array:
            v = inner_case.flowfield.vmap_call(x=x_, t=t_n)
            if self.free_wake:
                v += compute_v_ind(
                    cs=x_,
                    zetas=ArrayList([*zeta_b_nm1, *zeta_w_nm1]),
                    gammas=ArrayList([*gamma_b_nm1, *gamma_w_nm1]),
                    kernels=[*inner_case.kernels_b, *inner_case.kernels_w],
                    batch_size=self.batch_size,
                    mirror_normal=inner_case.mirror_normal,
                    mirror_point=inner_case.mirror_point,
                )
            return v

        zeta_w_nm1_update, gamma_w_nm1_update = propagate_wake(
            gamma_b_nm1=gamma_b_nm1,
            gamma_w_nm1=gamma_w_nm1,
            zeta_b_n=zeta_b_n,
            zeta_w_nm1=zeta_w_nm1,
            delta_w=inner_case.delta_w,
            v_func=v_wake_prop,
            dt=inner_case.dt,
            frozen_wake=False,
            linearise_variable_wake=False,
        )

        return (zeta_w_nm1_update - zeta_w_n).ravel(), (
            gamma_w_nm1_update - gamma_w_n
        ).ravel()

    # noinspection PyUnusedLocal
    def gamma_b_dot_res_func(
        self,
        gamma_b_nm1: Array,
        gamma_b_n: Array,
        gamma_b_dot_nm1: Array,
        gamma_b_dot_n: Array,
        dv: AeroelasticDesignVariables,
    ) -> Array:
        r"""
        Bound circulation time derivative residual.

        :math:`\frac{g}{h} \left[\mathbf{\Gamma}_{b, n} - \mathbf{\Gamma}_{b, n-1}\right] + (1-g)
        \dot{\mathbf{\Gamma}}_{b, n-1} - \dot{\mathbf{\Gamma}}_{b, n}`

        :param gamma_b_nm1: Bound circulation vector at timestep n-1.
        :param gamma_b_n: Bound circulation vector at timestep n.
        :param gamma_b_dot_nm1: Bound circulation time derivative vector at timestep n-1.
        :param gamma_b_dot_n: Bound circulation time derivative vector at timestep n.
        :param dv: Design variables. Whilst this function does not depend upon it, including it simplified obtaining
        the residual design gradient.
        :return: Bound circulation time derivative residual.
        """

        gamma_b_nm1 = ArrayList.from_vector(
            vect=gamma_b_nm1,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        gamma_b_n = ArrayList.from_vector(
            vect=gamma_b_n,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        gamma_b_dot_nm1 = ArrayList.from_vector(
            vect=gamma_b_dot_nm1,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        gamma_b_dot_n = ArrayList.from_vector(
            vect=gamma_b_dot_n,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        return (
            self.gamma_dot_relaxation / self.dt * (gamma_b_n - gamma_b_nm1)
            + (1.0 - self.gamma_dot_relaxation) * gamma_b_dot_nm1
            - gamma_b_dot_n
        ).ravel()

    def f_aero_res_func(
        self,
        i_ts: int,
        t_n: Array,
        varphi_n: Array,
        v_n: Array,
        gamma_b_n: Array,
        gamma_w_n: Array,
        gamma_b_dot_n: Array,
        zeta_w_n: Array,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        struct_obj: BeamStructure,
        f_aero_beam_n: Array,
        block_grid_gradients: bool,
        solve_dofs: tuple[int, ...],
    ) -> Array:
        r"""
        Aerodynamic forcing residual, compared in the local frame for compatibility with the structure.

        :math:`\boldsymbol{\mathcal{F}}_{\text{aero}, n}(\mathbf{\Gamma}_{b, n}, \mathbf{\Gamma}_{w, n},
        \dot{\mathbf{\Gamma}}_{b, n}, \boldsymbol{\zeta}_{b, n}, \dot{\boldsymbol{\zeta}}_{b, n},
        \boldsymbol{\zeta}_{w, n}) - \mathbf{f}_{\text{aero}, n}`

        :param i_ts: Time step index.
        :param t_n: Time at timestep n.
        :param varphi_n: Beam minimal coordinates vector at timestep n.
        :param v_n: Beam velocity vector at timestep n.
        :param gamma_b_n: Bound circulation vector at timestep n.
        :param gamma_w_n: Wake circulation vector at timestep n.
        :param gamma_b_dot_n: Bound circulation vector time derivative at timestep n.
        :param zeta_w_n: Wake grid vector at timestep n.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions.
        :param struct_obj: Beam structure.
        :param f_aero_beam_n: Local aerodynamic forcing projected onto beam.
        :param block_grid_gradients: If true, blocks the gradient path for the dependency of the steady aerodynamic
        forcing on the bound grid.
        :param solve_dofs: Index of forces to keep for solution.
        """

        varphi_n = varphi_n.reshape(-1, 6)
        v_n = v_n.reshape(-1, 6)

        gamma_b_n = ArrayList.from_vector(
            vect=gamma_b_n,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        gamma_w_n = ArrayList.from_vector(
            vect=gamma_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star, gd.n) for gd in self.grid_disc]
            ),
        )

        gamma_b_dot_n = ArrayList.from_vector(
            vect=gamma_b_dot_n,
            arr_list_shapes=ArrayListShape([(gd.m, gd.n) for gd in self.grid_disc]),
        )

        zeta_w_n = ArrayList.from_vector(
            vect=zeta_w_n,
            arr_list_shapes=ArrayListShape(
                [(gd.m_star + 1, gd.n + 1, 3) for gd in self.grid_disc]
            ),
        )

        f_aero_beam_n = f_aero_beam_n.reshape(-1, 6)

        inner_struct = struct_obj.case_from_dv(dv=dv.structure)
        hg_n = inner_struct.calculate_hg_from_varphi(varphi=varphi_n)
        hg_dot_n = inner_struct.make_hg_dot(hg=hg_n, v=v_n)

        inner_case = self.case_from_dv(dv=dv.aero)

        # create grid
        cs_ang_n, cs_vel_n = dv.aero.get_cs_n(i_ts=i_ts, dv_full=dv_full.aero)

        zeta_b_n = inner_case.hg_to_zeta_b(hg_n=hg_n, cs_ang_n=cs_ang_n)
        if block_grid_gradients:
            zeta_b_n = jax.lax.stop_gradient(zeta_b_n)
            zeta_w_n = jax.lax.stop_gradient(zeta_w_n)

        zeta_b_dot_n = inner_case.hg_dot_to_zeta_b_dot(
            hg_n=hg_n, hg_dot_n=hg_dot_n, cs_ang_n=cs_ang_n, cs_vel_n=cs_vel_n
        )
        nc_n = compute_nc(zetas=zeta_b_n)

        def v_total_func(x_: Array) -> Array:
            return inner_case.flowfield.vmap_call(x=x_, t=t_n) + compute_v_ind(
                cs=x_,
                zetas=ArrayList([*zeta_b_n, *zeta_w_n]),
                gammas=ArrayList([*gamma_b_n, *gamma_w_n]),
                kernels=[*inner_case.kernels_b, *inner_case.kernels_w],
                batch_size=self.batch_size,
                mirror_normal=inner_case.mirror_normal,
                mirror_point=inner_case.mirror_point,
            )

        f_steady = calculate_steady_forcing(
            zeta_b=zeta_b_n,
            zeta_dot_b=zeta_b_dot_n,
            gamma_b=gamma_b_n,
            gamma_w=gamma_w_n,
            rho=inner_case.flowfield.rho,
            v_func=v_total_func,
            v_inputs=None,
        )

        if self.include_unsteady_force:
            f_unsteady: Optional[ArrayList] = (
                ArrayList(
                    [
                        split_to_vertex(
                            inner_case.flowfield.rho
                            * gamma_b_dot_n[i_surf][..., None]
                            * nc_n[i_surf],
                            (0, 1),
                        )
                        for i_surf in range(inner_case.n_surf)
                    ]
                )
                if self.include_unsteady_force
                else None
            )
            f_tot = f_steady + f_unsteady
        else:
            f_tot = f_steady

        # project forcing to beam (global frame)
        f_tot_beam_global = project_forcing_to_beam(
            f_total=f_tot,
            rmat=hg_n[:, :3, :3],
            dof_mapping=inner_case.dof_mapping,
            x0_aero=inner_case.x0_b,
        )

        # transform to local frame to match f_aero_beam_n
        f_tot_beam_local = transform_nodal_vect(
            vect=f_tot_beam_global, rmat=jnp.swapaxes(hg_n[:, :3, :3], -2, -1)
        )

        return (f_tot_beam_local - f_aero_beam_n).ravel()[jnp.array(solve_dofs)]

    def timestep_residual(
        self,
        i_ts: int,
        varphi_nm1: Array,
        varphi_n: Array,
        v_n: Array,
        t_n: Array,
        q_n: AeroStates,
        q_nm1: AeroStates,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        f_aero_beam_n: Array,
        struct_obj: BeamStructure,
        approx_grads: bool,
    ) -> Array:
        r"""
        Compute the residual vector to the UVLM equations. These are given as:

        :math:`\left(\boldsymbol{\mathcal{A}}_{b, n} \cdot \mathbf{n}_n\right)^{-1} \left[\left(
        \boldsymbol{\mathcal{A}}_{w, n} \boldsymbol{\Gamma}_{w, n} +\mathbf{v}_{bc, n} - \dot{\mathbf{c}}\right)
        \cdot \mathbf{n}_n\right] + \boldsymbol{\Gamma}_{b, n}`

        :math:`\boldsymbol{\mathcal{W}}_{\Gamma}(\mathbf{\Gamma}_{b, {n-1}}, \mathbf{\Gamma}_{w, {n-1}})
        - \mathbf{\Gamma}_{w, n}`

        :math:`\frac{g}{h} \left[\mathbf{\Gamma}_{b, n} - \mathbf{\Gamma}_{b, n-1}\right] + (1-g)
        \dot{\mathbf{\Gamma}}_{b, n-1} - \dot{\mathbf{\Gamma}}_{b, n}`

        :math:`\boldsymbol{\mathcal{W}}_{\zeta}(\boldsymbol{\zeta}_{b, n}, \boldsymbol{\zeta}_{w, n-1})
        - \boldsymbol{\zeta}_{w, n}`

        :math:`\boldsymbol{\mathcal{F}}_{\text{aero}, n}(\mathbf{\Gamma}_{b, n}, \mathbf{\Gamma}_{w, n},
        \dot{\mathbf{\Gamma}}_{b, n}, \boldsymbol{\zeta}_{b, n}, \dot{\boldsymbol{\zeta}}_{b, n},
        \boldsymbol{\zeta}_{w, n}) - \mathbf{f}_{\text{aero}, n}`

        :param i_ts: Time step index.
        :param varphi_nm1: Beam minimal coordinates at timestep n-1, [n_nodes, 6].
        :param varphi_n: Beam maximal coordinates at timestep n, [n_nodes, 6].
        :param v_n: Beam velocity at timestep n, [n_nodes, 6].
        :param t_n: Time at step n.
        :param q_n: Aero minimal states at timestep n.
        :param q_nm1: Aero minimal states at timestep n-1.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions.
        :param f_aero_beam_n: Aerodynamic forcing for the beam at timestep n, in the local frame.
        :param struct_obj: Beam structure.
        :param approx_grads: If true, eliminate grid gradients from the aerodynamic force residual.
        :return: Residual vector.
        """

        zeta_w_res, gamma_w_res = self.wake_prop_res_func(
            i_ts=i_ts,
            varphi_n=varphi_n.ravel(),
            varphi_nm1=varphi_nm1.ravel(),
            t_n=t_n,
            dv=dv,
            dv_full=dv_full,
            gamma_b_nm1=q_nm1.gamma_b.ravel(),
            gamma_w_nm1=q_nm1.gamma_w.ravel(),
            gamma_w_n=q_n.gamma_w.ravel(),
            zeta_w_nm1=q_nm1.zeta_w.ravel(),
            zeta_w_n=q_n.zeta_w.ravel(),
            struct_obj=struct_obj,
        )

        return jnp.concatenate(
            (
                self.gamma_b_res_func(
                    i_ts=i_ts,
                    varphi_n=varphi_n.ravel(),
                    v_n=v_n.ravel(),
                    t_n=t_n,
                    dv=dv,
                    dv_full=dv_full,
                    gamma_b_n=q_n.gamma_b.ravel(),
                    gamma_w_n=q_n.gamma_w.ravel(),
                    zeta_w_n=q_n.zeta_w.ravel(),
                    struct_obj=struct_obj,
                ),
                gamma_w_res,
                self.gamma_b_dot_res_func(
                    gamma_b_nm1=q_nm1.gamma_b.ravel(),
                    gamma_b_n=q_n.gamma_b.ravel(),
                    gamma_b_dot_nm1=q_nm1.gamma_b_dot.ravel(),
                    gamma_b_dot_n=q_n.gamma_b_dot.ravel(),
                    dv=dv,
                ),
                zeta_w_res,
                self.f_aero_res_func(
                    i_ts=i_ts,
                    varphi_n=varphi_n.ravel(),
                    v_n=v_n.ravel(),
                    t_n=t_n,
                    dv=dv,
                    dv_full=dv_full,
                    gamma_b_n=q_n.gamma_b.ravel(),
                    gamma_w_n=q_n.gamma_w.ravel(),
                    gamma_b_dot_n=q_n.gamma_b_dot.ravel(),
                    zeta_w_n=q_n.zeta_w.ravel(),
                    f_aero_beam_n=f_aero_beam_n.ravel(),
                    struct_obj=struct_obj,
                    block_grid_gradients=approx_grads,
                    solve_dofs=tuple(range(struct_obj.n_dof)),
                ),
            )
        )

    def construct_approximate_jacobians(
        self,
        aero_sol: DynamicAeroCase,
        structure_sol: DynamicStructure,
        struct_obj: BeamStructure,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        solve_dofs: tuple[int, ...],
        jacobian_approximations: AeroJacobianApproximations,
    ) -> dict[str, dict[str, Optional[Callable[..., Any]]]]:
        r"""
        Compute approximations for the aerodynamic residual Jacobians specified in the ``jacobian_approximations`` data
        structure. The residuals covered are the bound circulation, wake propagation, bound circulation rate, and
        aerodynamic forcing.
        :param aero_sol: Aerodynamic solution from which to extract states for the initial time step.
        :param structure_sol: Structural solution from which to extract states for the initial time step.
        :param struct_obj: Beam structure used by the residual functions for the kinematic transformations.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions for the variables where gradients aren't
        requested.
        :param solve_dofs: Active structural degrees of freedom.
        :param jacobian_approximations: Data structure which defines which approximations to create.
        :return: Dictionary of approximations keyed by residual name.
        """
        q_nm1 = aero_sol.get_states(0)
        q_n = aero_sol.get_states(1)
        struct_nm1 = structure_sol.get_minimal_states(0)
        struct_n = structure_sol.get_minimal_states(1)

        varphi_nm1 = struct_nm1.varphi.ravel()
        varphi_n = struct_n.varphi.ravel()
        v_n = struct_n.v.ravel()
        gamma_b_nm1 = q_nm1.gamma_b.ravel()
        gamma_b_n = q_n.gamma_b.ravel()
        gamma_w_nm1 = q_nm1.gamma_w.ravel()
        gamma_w_n = q_n.gamma_w.ravel()
        gamma_b_dot_nm1 = q_nm1.gamma_b_dot.ravel()
        gamma_b_dot_n = q_n.gamma_b_dot.ravel()
        zeta_w_nm1 = q_nm1.zeta_w.ravel()
        zeta_w_n = q_n.zeta_w.ravel()

        if struct_n.f_ext_aero is None:
            raise ValueError("Missing aerodynamic forcing states")
        f_aero_beam_n = struct_n.f_ext_aero.ravel()

        t_n = aero_sol.t[1]

        gamma_b_args = dict(
            i_ts=1,
            t_n=t_n,
            varphi_n=varphi_n,
            v_n=v_n,
            gamma_b_n=gamma_b_n,
            gamma_w_n=gamma_w_n,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
        )

        wake_args = dict(
            i_ts=1,
            t_n=t_n,
            varphi_nm1=varphi_nm1,
            varphi_n=varphi_n,
            gamma_b_nm1=gamma_b_nm1,
            gamma_w_nm1=gamma_w_nm1,
            gamma_w_n=gamma_w_n,
            zeta_w_nm1=zeta_w_nm1,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
        )

        gamma_b_dot_args = dict(
            gamma_b_nm1=gamma_b_nm1,
            gamma_b_n=gamma_b_n,
            gamma_b_dot_nm1=gamma_b_dot_nm1,
            gamma_b_dot_n=gamma_b_dot_n,
            dv=dv,
        )

        f_aero_args = dict(
            i_ts=1,
            t_n=t_n,
            varphi_n=varphi_n,
            v_n=v_n,
            gamma_b_n=gamma_b_n,
            gamma_w_n=gamma_w_n,
            gamma_b_dot_n=gamma_b_dot_n,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
            f_aero_beam_n=f_aero_beam_n,
            block_grid_gradients=True,
            solve_dofs=solve_dofs,
        )

        res_args: dict[
            str, tuple[Callable[..., Array], dict[str, Any], Sequence[str]]
        ] = {
            "gamma_b": (
                self.gamma_b_res_func,
                gamma_b_args,
                [f.name for f in fields(GammaBApprox)],
            ),
            "gamma_w": (
                lambda **kwargs: self.wake_prop_res_func(**kwargs)[1],
                wake_args,
                [f.name for f in fields(GammaWApprox)],
            ),
            "zeta_w": (
                lambda **kwargs: self.wake_prop_res_func(**kwargs)[0],
                wake_args,
                [f.name for f in fields(ZetaWApprox)],
            ),
            "gamma_b_dot": (
                self.gamma_b_dot_res_func,
                gamma_b_dot_args,
                [f.name for f in fields(GammaBDotApprox)],
            ),
            "f_aero": (
                self.f_aero_res_func,
                f_aero_args,
                [f.name for f in fields(FAeroApprox)],
            ),
        }

        return construct_approximation(
            res_args=res_args, jacobian_approximations=jacobian_approximations
        )

    def timestep_residual_jacobians(
        self,
        i_ts: int | Array,
        varphi_nm1: Array,
        varphi_n: Array,
        v_n: Array,
        t_n: Array,
        q_n: AeroStates,
        q_nm1: AeroStates,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        f_aero_beam_n: Array,
        struct_obj: BeamStructure,
        approx_grads: bool,
        solve_dofs: tuple[int, ...],
        n_profile_loops: Optional[int],
        jac_options: dict,
    ) -> tuple[
        Array,
        Array,
        AeroelasticDesignVariables,
        Array,
        Array,
        Optional[dict[str, dict[str, float]]],
        Optional[dict[str, dict[str, float]]],
    ]:
        r"""
        Compute the Jacobians of the aerodynamic problem.
        :param i_ts: Time step index.
        :param varphi_nm1: Minimal structural coordinates at timestep n-1, [n_nodes, 6].
        :param varphi_n: Minimal structural coordinates at timestep n, [n_nodes, 6].
        :param v_n: Structural velocity at timestep n, [n_nodes, 6].
        :param t_n: Time at timestep n.
        :param q_n: Aerodynamic minimal states at timestep n.
        :param q_nm1: Aerodynamic minimal states at timestep n-1.
        :param dv: Aeroelastic design variables.
        :param dv_full: Aeroelastic design variables without omissions.
        :param f_aero_beam_n: Aerodynamic forcing in local frame of reference at timestep n, [n_nodes, 6].
        :param struct_obj: Structural object.
        :param approx_grads: If true, eliminate grid gradients from force computation.
        :param solve_dofs: Degrees of freedom to solve for. This removes non-active forcing entries.
        :param n_profile_loops: Optional number of loops for profiling function.
        :param jac_options: Dictionary of optional functions which can be used to substitute Jacobian evaluations from
        AD.
        :return: Gradients of aerodynamic residual with respect to previous states, current states, and design
        variables. Additionally, includes Jacobians of the aerodynamic residual with respect to the structural
        displacement and velocity, and profiling times for compilation and run time.
        """

        compile_time: dict[str, dict[str, float]] = {}
        run_time: dict[str, dict[str, float]] = {}

        varphi_nm1 = varphi_nm1.ravel()
        varphi_n = varphi_n.ravel()
        v_n = v_n.ravel()
        gamma_b_nm1 = q_nm1.gamma_b.ravel()
        gamma_b_n = q_n.gamma_b.ravel()
        gamma_w_nm1 = q_nm1.gamma_w.ravel()
        gamma_w_n = q_n.gamma_w.ravel()
        gamma_b_dot_nm1 = q_nm1.gamma_b_dot.ravel()
        gamma_b_dot_n = q_n.gamma_b_dot.ravel()
        zeta_w_nm1 = q_nm1.zeta_w.ravel()
        zeta_w_n = q_n.zeta_w.ravel()

        d_gamma_b, compile_time["gamma_b"], run_time["gamma_b"] = jacrev_custom(
            func=self.gamma_b_res_func,
            jac_options=jac_options["gamma_b"],
            n_profile_loops=n_profile_loops,
            func_name="gamma_b",
        )(
            i_ts=i_ts,
            t_n=t_n,
            varphi_n=varphi_n,
            v_n=v_n,
            gamma_b_n=gamma_b_n,
            gamma_w_n=gamma_w_n,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
        )

        d_gamma_w, compile_time["gamma_w"], run_time["gamma_w"] = jacrev_custom(
            func=lambda **kwargs: self.wake_prop_res_func(**kwargs)[1],
            jac_options=jac_options["gamma_w"],
            n_profile_loops=n_profile_loops,
            func_name="gamma_w",
        )(
            i_ts=i_ts,
            t_n=t_n,
            varphi_nm1=varphi_nm1,
            varphi_n=varphi_n,
            gamma_b_nm1=gamma_b_nm1,
            gamma_w_nm1=gamma_w_nm1,
            gamma_w_n=gamma_w_n,
            zeta_w_nm1=zeta_w_nm1,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
        )

        d_zeta_w, compile_time["zeta_w"], run_time["zeta_w"] = jacrev_custom(
            func=lambda **kwargs: self.wake_prop_res_func(**kwargs)[0],
            jac_options=jac_options["zeta_w"],
            n_profile_loops=n_profile_loops,
            func_name="zeta_w",
        )(
            i_ts=i_ts,
            t_n=t_n,
            varphi_nm1=varphi_nm1,
            varphi_n=varphi_n,
            gamma_b_nm1=gamma_b_nm1,
            gamma_w_nm1=gamma_w_nm1,
            gamma_w_n=gamma_w_n,
            zeta_w_nm1=zeta_w_nm1,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
        )

        d_gamma_b_dot, compile_time["gamma_b_dot"], run_time["gamma_b_dot"] = (
            jacrev_custom(
                func=self.gamma_b_dot_res_func,
                jac_options=jac_options["gamma_b_dot"],
                n_profile_loops=n_profile_loops,
                func_name="gamma_b_dot",
            )(
                gamma_b_nm1=gamma_b_nm1,
                gamma_b_n=gamma_b_n,
                gamma_b_dot_nm1=gamma_b_dot_nm1,
                gamma_b_dot_n=gamma_b_dot_n,
                dv=dv,
            )
        )

        d_f_aero, compile_time["f_aero"], run_time["f_aero"] = jacrev_custom(
            func=self.f_aero_res_func,
            jac_options=jac_options["f_aero"],
            n_profile_loops=n_profile_loops,
            func_name="f_aero",
            static_argnames=("block_grid_gradients", "solve_dofs"),
        )(
            i_ts=i_ts,
            t_n=t_n,
            varphi_n=varphi_n,
            v_n=v_n,
            gamma_b_n=gamma_b_n,
            gamma_w_n=gamma_w_n,
            gamma_b_dot_n=gamma_b_dot_n,
            zeta_w_n=zeta_w_n,
            dv=dv,
            dv_full=dv_full,
            struct_obj=struct_obj,
            f_aero_beam_n=f_aero_beam_n.ravel(),
            block_grid_gradients=approx_grads,
            solve_dofs=solve_dofs,
        )
        # slice f_aero Jacobian to solve_dofs columns (input is full n_dof, state is n_solve)
        d_f_aero["f_aero_beam_n"] = d_f_aero["f_aero_beam_n"][:, jnp.array(solve_dofs)]

        # Jacobians block widths and heights
        n_solve_dof = len(solve_dofs)
        aero_sizes = (
            gamma_b_n.size,
            gamma_w_n.size,
            gamma_b_n.size,
            zeta_w_n.size,
            n_solve_dof,
        )

        struct_sizes = (
            struct_obj.n_dof,
            struct_obj.n_dof,
            struct_obj.n_dof,
            struct_obj.n_dof,
        )

        d_aero_res_d_q_nm1 = construct_named_block_jacobian(
            entries=(d_gamma_b, d_gamma_w, d_gamma_b_dot, d_zeta_w, d_f_aero),
            keys=(
                "gamma_b_nm1",
                "gamma_w_nm1",
                "gamma_b_dot_nm1",
                "zeta_w_nm1",
                "f_aero_beam_nm1",
            ),
            widths=aero_sizes,
            heights=aero_sizes,
        )

        d_aero_res_d_q_n = construct_named_block_jacobian(
            entries=(d_gamma_b, d_gamma_w, d_gamma_b_dot, d_zeta_w, d_f_aero),
            keys=(
                "gamma_b_n",
                "gamma_w_n",
                "gamma_b_dot_n",
                "zeta_w_n",
                "f_aero_beam_n",
            ),
            widths=aero_sizes,
            heights=aero_sizes,
        )

        # residual of aero problem w.r.t. structural states
        d_struct_res_d_q_nm1 = construct_named_block_jacobian(
            entries=(d_gamma_b, d_gamma_w, d_gamma_b_dot, d_zeta_w, d_f_aero),
            keys=("varphi_nm1", "v_nm1", "v_dot_nm1", "a_nm1"),
            widths=struct_sizes,
            heights=aero_sizes,
        )

        d_struct_res_d_q_n = construct_named_block_jacobian(
            entries=(d_gamma_b, d_gamma_w, d_gamma_b_dot, d_zeta_w, d_f_aero),
            keys=("varphi_n", "v_n", "v_dot_n", "a_n"),
            widths=struct_sizes,
            heights=aero_sizes,
        )

        # handle design gradients
        d_res_d_dv = AeroelasticDesignVariables.concatenate(
            d_gamma_b["dv"],
            d_gamma_w["dv"],
            d_gamma_b_dot["dv"],
            d_zeta_w["dv"],
            d_f_aero["dv"],
        )

        return (
            d_aero_res_d_q_nm1,
            d_aero_res_d_q_n,
            d_res_d_dv,
            d_struct_res_d_q_nm1,
            d_struct_res_d_q_n,
            compile_time if n_profile_loops is not None else None,
            run_time if n_profile_loops is not None else None,
        )

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
            "free_wake",
            "gamma_dot_relaxation",
            "include_unsteady_force",
            "grid_func",
            "batch_size",
        )

    @staticmethod
    def _dynamic_names() -> Sequence[str]:

        return (
            "_hg0",
            "_x0_b",
            "_zeta0_b",
            "_zeta0_w",
            "_dt",
            "_flowfield",
            "_delta_w",
            "mirror_point",
            "mirror_normal",
            "cs_ang0",
            "cs_vel0",
        )
