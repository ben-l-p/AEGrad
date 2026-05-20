from __future__ import annotations

from copy import deepcopy
from typing import Optional, Callable, overload, cast

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.utils.print_utils import jax_print, VerbosityLevel
from aegrad.structure.beam import BaseBeamStructure
from aegrad.structure import OptionalJacobians
from aegrad.structure.data_structures import (
    StaticStructure,
)
from aegrad.structure.gradients.data_structures import (
    StructureFullStates,
    StructuralDesignVariables,
    StructuralGradsToCompute,
)
from aegrad.algebra.se3 import exp_se3, log_se3
from aegrad.structure import DynamicStructure
from aegrad.structure.data_structures import StructureMinimalStates
from aegrad.structure.utils import get_solve_dofs, transform_nodal_vect
from aegrad.algebra.array_utils import construct_named_block_jacobian
from aegrad.utils.utils import make_pytree

type StructuralObjectiveFunction = Callable[
    [StructureFullStates, StructuralDesignVariables, Optional[int | Array]], Array
]


@make_pytree
class BeamStructure(BaseBeamStructure):
    def case_from_dv(self, dv: StructuralDesignVariables) -> BeamStructure:
        r"""
        Obtain a structural object as a function of design variables, allowing it to have defined gradients w.r.t. design variables.
        :param dv: Design variables.
        :return: Beam structure object with the same functionality as self.
        """
        inner_case = deepcopy(self)
        inner_case.set_design_variables(
            coords=dv.x0 if dv.x0 is not None else self.x0,
            k_cs=dv.k_cs if dv.k_cs is not None else self.k_cs,
            m_cs=dv.m_cs if dv.m_cs is not None else self.m_cs,
            m_lumped=dv.m_lumped if dv.m_lumped is not None else self._m_lumped,
            remove_checks=True,
        )

        return inner_case

    def minimal_states_to_full_states(
        self,
        i_ts: int,
        q: StructureMinimalStates,
        dv: StructuralDesignVariables,
        dv_full: StructuralDesignVariables,
    ) -> StructureFullStates:
        r"""
        Obtain the full set of states from the minimal states and the design variables.
        :param i_ts: Index of the time step.
        :param q: Minimal dynamic structure states.
        :param dv: Design variables, where entries for gradients which aren't needed are set to None.
        :param dv_full: Design variables, without omissions. These values are fallen back to when an entry in ``dv`` is
        none, to give an equivalent with zero gradient.
        :return: Full set of structural states used inside objective function.
        """
        struct = self.case_from_dv(dv)
        hg = struct.calculate_hg_from_varphi(q.varphi)
        d = struct.make_d(hg=hg)
        p_d = struct.make_p_d(d=d)
        eps = struct.make_eps(d=d)
        f_elem = struct.make_f_elem(eps=eps)

        assert dv_full.thrust_t is not None
        f_res = struct.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.f_ext_follower[i_ts, ...]
            if dv.f_ext_follower is not None
            else dv_full.f_ext_follower[i_ts, ...]
            if dv_full.f_ext_follower is not None
            else None,
            f_ext_dead_n=dv.f_ext_dead[i_ts, ...]
            if dv.f_ext_dead is not None
            else dv_full.f_ext_dead[i_ts, ...]
            if dv_full.f_ext_dead is not None
            else None,
            thrust_n={k: v[i_ts] for k, v in dv.thrust_t.items()}
            if dv.thrust_t is not None
            else {k: v[i_ts] for k, v in dv_full.thrust_t.items()},
            dynamic=True,
            m_t=self.make_m_t(d=d),
            c_l=self._make_c_t(d=d, d_dot=self._make_d_dot(p_d=p_d, v=q.v), v=q.v)[0],
            c_l_lumped=self._make_c_t_lumped(v=q.v)[0]
            if self.use_lumped_mass
            else None,
            v=q.v,
            v_dot=q.v_dot,
        )[0]
        return StructureFullStates(
            v=q.v, v_dot=q.v_dot, eps=eps, hg=hg, f_elem=f_elem, f_res=f_res
        )

    def _structural_states_res_from_dv_varphi(
        self,
        dv: StructuralDesignVariables,
        varphi: Array,
        thrust: dict[str, Array],
    ) -> StructureFullStates:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        :param dv: Design variables.
        :param varphi: Twist coordinates which map from the reference configuration to the current as
        :math:`\mathbf{H} = \mathbf{H}_0 \mathrm{exp} (\varphi)`.
        :param thrust: Thrust. This is only needed when thrust is not included as a design variable. {keys, [1]}.
        :return: Structural states and forcing residual.
        """

        inner_case = self.case_from_dv(dv=dv)

        hg = inner_case.calculate_hg_from_varphi(varphi=varphi)  # [n_nodes_, 4, 4]
        d = inner_case.make_d(hg)
        p_d = inner_case.make_p_d(d)
        eps = inner_case.make_eps(d)
        f_elem = jnp.einsum("ijk,ik->ij", inner_case.k_cs, eps)

        if inner_case.use_gravity:
            m_t = inner_case.make_m_t(d)
        else:
            m_t = None

        f_res = inner_case.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.f_ext_follower,
            f_ext_dead_n=dv.f_ext_dead,
            thrust_n=dv.thrust_t if dv.thrust_t is not None else thrust,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        return StructureFullStates(
            hg=hg,
            eps=eps,
            f_elem=f_elem,
            f_res=f_res,
            v=None,
            v_dot=None,
        )

    def static_adjoint(
        self,
        structure: StaticStructure,
        objective: StructuralObjectiveFunction,
        optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
            True, True, True, True
        ),
        forward_adjoint: bool = False,
    ) -> tuple[StructuralDesignVariables, Array]:
        r"""
        Computes the static grads of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param structure: StaticStructure containing the current state of the structure.
        :param objective: Objective function that takes the structure and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :param forward_adjoint: Flag on which to use of the forward or reverse adjoint.
        :return: Gradient of objective function output with respect to design variables, and adjoint states.
        """

        solve_dofs = jnp.array(
            get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs)
        )

        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # Recover original global dead force: structure.f_ext_dead is stored in local frame as
        # f_local = R^T @ f_global, so f_global = R @ f_local
        rmat = structure.hg[:, :3, :3]
        f_ext_dead_global = (
            transform_nodal_vect(structure.f_ext_dead, rmat)
            if structure.f_ext_dead is not None
            else None
        )

        # make design variables for current state of structure
        dv = StructuralDesignVariables(
            x0=self.x0,
            orientation_euler=self.orientation_euler,
            k_cs=self.k_cs,
            m_cs=self._m_cs,
            m_lumped=self.m_lumped if self.use_lumped_mass else None,
            f_ext_follower=structure.f_ext_follower,
            f_ext_dead=f_ext_dead_global,
            thrust_t=structure.thrust,
            f_shape=(),
        )

        struct_states = structure.get_full_states()

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv, None))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)
        n_u_full = self.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n, p_f_p_x = jax.jacrev(
            lambda varphi_, dv_: objective(
                self._structural_states_res_from_dv_varphi(
                    dv=dv_, varphi=varphi_, thrust=structure.thrust
                ),
                dv_,
                None,
            ),
            argnums=(0, 1),
        )(structure.varphi, dv)

        p_f_p_n = p_f_p_n.reshape(n_f, n_u_full)[:, solve_dofs]  # [n_f, n_u]
        p_f_p_x = p_f_p_x.ravel_jacobian(n_f, n_x)  # [n_f, n_x]

        # gradient of residual w.r.t. design variables and minimal states
        p_res_p_x, p_res_p_varphi = (jax.jacfwd if n_u > n_x else jax.jacrev)(
            lambda dv_, varphi_: (
                self._structural_states_res_from_dv_varphi(
                    dv=dv_, varphi=varphi_, thrust=structure.thrust
                ).f_res
            ),
            argnums=(0, 1),
        )(dv, structure.varphi)

        p_res_p_x = p_res_p_x.ravel_jacobian(n_u_full, n_x)[solve_dofs, :]  # [n_u, n_x]
        p_res_p_varphi = p_res_p_varphi.reshape(n_u_full, n_u_full)[
            jnp.ix_(solve_dofs, solve_dofs)
        ]  # [n_u, n_u]

        if forward_adjoint:
            # forward mode
            adj = jnp.linalg.solve(p_res_p_varphi, p_res_p_x).reshape(
                self.n_nodes, 6, n_x
            )  # [n_u, n_x]
            rhs = jnp.einsum("ij,ijk->k", p_f_p_n, adj)  # [n_f, n_x]
        else:
            # reverse mode
            adj = jnp.linalg.solve(p_res_p_varphi.T, p_f_p_n.T).T  # [n_f, n_u]
            rhs = adj @ p_res_p_x  # [n_f, n_x]

        return StructuralDesignVariables(
            **dv.from_adjoint(f_shape, p_f_p_x - rhs), f_shape=f_shape
        ), adj

    def varphi_res_func(
        self,
        varphi_nm1: Array,
        varphi_n: Array,
        v_nm1: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:

        varphi_nm1 = varphi_nm1.reshape(-1, 6)
        varphi_n = varphi_n.reshape(-1, 6)
        v_nm1 = v_nm1.reshape(-1, 6)
        a_nm1 = a_nm1.reshape(-1, 6)
        a_n = a_n.reshape(-1, 6)

        # time integrator parameters
        dt = self.time_integrator.dt
        beta = self.time_integrator.beta

        phi_n = dt * v_nm1 + (0.5 - beta) * dt * dt * a_nm1 + beta * dt * dt * a_n
        return vmap(
            lambda vp_n, vp_nm1, phi: log_se3(
                exp_se3(-vp_n) @ exp_se3(vp_nm1) @ exp_se3(phi)
            ),
            0,
            0,
            0,
        )(varphi_n, varphi_nm1, phi_n).ravel()[jnp.array(solve_dofs)]

    def v_res_func(
        self,
        v_nm1: Array,
        v_n: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:

        # time integrator parameters
        dt = self.time_integrator.dt
        gamma = self.time_integrator.gamma

        return (v_nm1 + (1.0 - gamma) * dt * a_nm1 + gamma * dt * a_n - v_n).ravel()[
            jnp.array(solve_dofs)
        ]

    def a_res_func(
        self,
        v_dot_nm1: Array,
        v_dot_n: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:

        # time integrator parameters
        alpha_f = self.time_integrator.alpha_f
        alpha_m = self.time_integrator.alpha_m

        return (
            ((1.0 - alpha_f) * v_dot_n + alpha_f * v_dot_nm1 - alpha_m * a_nm1)
            / (1.0 - alpha_m)
            - a_n
        ).ravel()[jnp.array(solve_dofs)]

    def v_dot_res_func(
        self,
        i_ts: int,
        varphi_nm1: Array,
        varphi_n: Array,
        v_nm1: Array,
        v_n: Array,
        v_dot_nm1: Array,
        v_dot_n: Array,
        dv: StructuralDesignVariables,
        f_ext_aero_nm1: Optional[Array],
        f_ext_aero_n: Optional[Array],
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> Array:

        varphi_nm1 = varphi_nm1.reshape(-1, 6)
        varphi_n = varphi_n.reshape(-1, 6)
        v_nm1 = v_nm1.reshape(-1, 6)
        v_n = v_n.reshape(-1, 6)
        v_dot_nm1 = v_dot_nm1.reshape(-1, 6)
        v_dot_n = v_dot_n.reshape(-1, 6)
        f_ext_aero_nm1 = (
            f_ext_aero_nm1.reshape(-1, 6) if f_ext_aero_nm1 is not None else None
        )
        f_ext_aero_n = f_ext_aero_n.reshape(-1, 6) if f_ext_aero_n is not None else None

        solve_dofs: Array = jnp.array(solve_dofs)

        # time integrator parameters
        alpha_f = self.time_integrator.alpha_f

        # updates to v_dot, which are obtained from relation to other states through structural problem
        inner_case = self.case_from_dv(
            dv=dv
        )  # allows for gradients w.r.t. design variables

        varphi_alpha = inner_case.time_integrator.calculate_varphi_alpha(
            varphi_nm1=varphi_nm1, varphi_n=varphi_n
        )
        v_alpha = inner_case.time_integrator.calculate_v_alpha(v_nm1=v_nm1, v_n=v_n)
        v_dot_alpha = inner_case.time_integrator.calculate_v_dot_alpha(
            v_dot_nm1=v_dot_nm1, v_dot_n=v_dot_n
        )

        hg_alpha = inner_case.calculate_hg_from_varphi(varphi_alpha)

        # obtain forces at alpha
        f_ext_dead_alpha = (
            inner_case.time_integrator.calculate_f_alpha(
                f_nm1=dv.f_ext_dead[i_ts - 1, ...], f_n=dv.f_ext_dead[i_ts, ...]
            )
            if dv.f_ext_dead is not None
            else None
        )
        f_ext_follower_alpha = (
            inner_case.time_integrator.calculate_f_alpha(
                f_nm1=dv.f_ext_follower[i_ts - 1, ...],
                f_n=dv.f_ext_follower[i_ts, ...],
            )
            if dv.f_ext_follower is not None
            else None
        )

        if f_ext_aero_n is not None and f_ext_aero_nm1 is not None:
            f_aero_alpha = inner_case.time_integrator.calculate_f_alpha(
                f_nm1=f_ext_aero_nm1, f_n=f_ext_aero_n
            )
        else:
            f_aero_alpha = None

        if dv.thrust_t is not None:
            thrust_t_: dict[str, Array] = dv.thrust_t
        else:
            # if not included as a design variable, use the input value
            thrust_t_ = thrust_t

        thrust_alpha: dict[str, Array] = {
            k: inner_case.time_integrator.calculate_f_alpha(
                f_nm1=v[i_ts - 1], f_n=v[i_ts, ...]
            )
            for k, v in thrust_t_.items()
        }

        # find system properties at alpha
        (
            d_alpha,
            _,
            f_dead_alpha,  # dead force in local frame
            _,
            f_grav_alpha,
            f_int_alpha,
            f_gyr_alpha,
            f_iner_alpha,
            _,
        ) = inner_case.resolve_forces(
            hg=hg_alpha,
            dynamic=True,
            f_ext_follower=f_ext_follower_alpha,
            f_ext_dead=f_ext_dead_alpha,
            f_ext_aero=None,  # this is None here, as we already have the aero force in the local frame
            thrust=thrust_alpha,
            v=v_alpha,
            v_dot=v_dot_alpha,
            approx_gradients=approx_grads,
        )

        # use stop gradient to prevent effective stiffness contribution
        m_alpha = inner_case.assemble_matrix_from_entries(
            inner_case.make_m_t(
                d=jax.lax.stop_gradient(d_alpha) if approx_grads else d_alpha
            )
        )
        if inner_case.use_lumped_mass:
            m_alpha = inner_case.add_lumped_contributions_to_arr(
                arr=m_alpha, lumped_arr=inner_case.m_lumped
            )

        # calculate non-inertial forcing residual
        f_res_non_iner = f_int_alpha + f_gyr_alpha
        if inner_case.use_gravity:
            f_res_non_iner += f_grav_alpha
        if (
            f_dead_alpha is not None
        ):  # use the output from the resolve forces function, as it's in the local frame
            f_res_non_iner += f_dead_alpha
        if f_ext_follower_alpha is not None:
            f_res_non_iner += f_ext_follower_alpha
        if f_aero_alpha is not None:
            f_res_non_iner += f_aero_alpha

        # from forcing residual, solve for v_dot which satisfies f_res=0
        # restrict solve to free DOFs to avoid prescribed-DOF reaction forces
        m_alpha = m_alpha[jnp.ix_(solve_dofs, solve_dofs)]

        # find the v_dot residual to satisfy the structural problem
        return f_res_non_iner.ravel()[solve_dofs] / (1.0 - alpha_f) - m_alpha @ (
            alpha_f / (1.0 - alpha_f) * v_dot_nm1.ravel()[solve_dofs]
            + v_dot_n.ravel()[solve_dofs]
        )

    def timestep_residual(
        self,
        i_ts: int,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        dv_: StructuralDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> Array:
        r"""
        Routine to compute the full residual for the structural dynamic problem.
        :param i_ts: Time step index.
        :param q_nm1: Previous minimal state.
        :param q_n: Current minimal state.
        :param dv_: Design variables.
        :param thrust_t: Thrust time history, {key, [n_tstep]}.
        :param solve_dofs: Solve degrees of freedom.
        :param approx_grads: If true, block gradients from some parts of the solution.
        :return: Residual vector, [4 * n_solve_dof].
        """
        return jnp.stack(
            (
                self.varphi_res_func(
                    varphi_nm1=q_nm1.varphi,
                    varphi_n=q_n.varphi,
                    v_nm1=q_nm1.v,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
                self.v_res_func(
                    v_nm1=q_nm1.v,
                    v_n=q_n.v,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
                self.v_dot_res_func(
                    i_ts=i_ts,
                    varphi_nm1=q_nm1.varphi,
                    varphi_n=q_n.varphi,
                    v_nm1=q_nm1.v,
                    v_n=q_n.v,
                    v_dot_nm1=q_nm1.v_dot,
                    v_dot_n=q_n.v_dot,
                    approx_grads=approx_grads,
                    f_ext_aero_nm1=q_nm1.f_ext_aero,
                    f_ext_aero_n=q_n.f_ext_aero,
                    thrust_t=thrust_t,
                    dv=dv_,
                    solve_dofs=solve_dofs,
                ),
                self.a_res_func(
                    v_dot_nm1=q_nm1.v_dot,
                    v_dot_n=q_n.v_dot,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
            ),
            axis=0,
        ).ravel()  # [4*n_free_dof]

    @overload
    def timestep_residual_jacobians(
        self,
        i_ts: int,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        f_ext_aero_nm1: Array,
        f_ext_aero_n: Array,
        dv: StructuralDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> tuple[Array, Array, StructuralDesignVariables, Array, Array]: ...

    @overload
    def timestep_residual_jacobians(
        self,
        i_ts: int,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        f_ext_aero_nm1: None,
        f_ext_aero_n: None,
        dv: StructuralDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> tuple[Array, Array, StructuralDesignVariables, None, None]: ...

    @jax.jit(static_argnums=(0, 6, 8, 9))
    def timestep_residual_jacobians(
        self,
        i_ts: int,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        f_ext_aero_nm1: Optional[Array],
        f_ext_aero_n: Optional[Array],
        dv: StructuralDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> tuple[
        Array, Array, StructuralDesignVariables, Optional[Array], Optional[Array]
    ]:
        r"""
        Obtain the Jacobians of the structural residual with respect to the current states and previous states.
        :param i_ts: Time step index.
        :param q_nm1: Previous minimal states.
        :param q_n: Current minimal states.
        :param f_ext_aero_nm1: Optional aerodynamic forcing for previous time step, [n_nodes, 6].
        :param f_ext_aero_n: Optional aerodynamic forcing for current time step, [n_nodes, 6].
        :param dv: Design variables.
        :param thrust_t: Thrust time history, {key, [n_tstep]}.
        :param solve_dofs: Index of degrees of freedom to solve for.
        :param approx_grads: If True, remove some gradient terms which are generally small.
        :return: Jacobians with respect to previous state and current state.
        """

        # varphi
        d_varphi = dict()
        (
            d_varphi["varphi_nm1"],
            d_varphi["varphi_n"],
            d_varphi["v_nm1"],
            d_varphi["a_nm1"],
            d_varphi["a_n"],
        ) = jax.jacrev(self.varphi_res_func, argnums=(0, 1, 2, 3, 4))(
            q_nm1.varphi.ravel(),
            q_n.varphi.ravel(),
            q_nm1.v.ravel(),
            q_nm1.a.ravel(),
            q_n.a.ravel(),
            solve_dofs,
        )

        # velocity
        d_v = dict()
        d_v["v_nm1"], d_v["v_n"], d_v["a_nm1"], d_v["a_n"] = jax.jacrev(
            self.v_res_func, argnums=(0, 1, 2, 3)
        )(q_nm1.v.ravel(), q_n.v.ravel(), q_nm1.a.ravel(), q_n.a.ravel(), solve_dofs)

        # acceleration (aero_dv Jacobian computed separately via VJP in the adjoint loop)
        d_v_dot = dict()
        compute_f_aero_grads = f_ext_aero_nm1 is not None and f_ext_aero_n is not None

        (
            d_v_dot["varphi_nm1"],
            d_v_dot["varphi_n"],
            d_v_dot["v_nm1"],
            d_v_dot["v_n"],
            d_v_dot["v_dot_nm1"],
            d_v_dot["v_dot_n"],
            p_v_dot_p_dv,
            *f_jacs,
        ) = jax.jacrev(
            self.v_dot_res_func,
            argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9)
            if compute_f_aero_grads
            else (1, 2, 3, 4, 5, 6, 7),
        )(
            i_ts,
            q_nm1.varphi.ravel(),
            q_n.varphi.ravel(),
            q_nm1.v.ravel(),
            q_n.v.ravel(),
            q_nm1.v_dot.ravel(),
            q_n.v_dot.ravel(),
            dv,
            f_ext_aero_nm1.ravel() if compute_f_aero_grads else None,  # type: ignore
            f_ext_aero_n.ravel() if compute_f_aero_grads else None,  # type: ignore
            thrust_t,
            solve_dofs,
            approx_grads,
        )

        if compute_f_aero_grads:
            p_v_dot_p_f_ext_nm1, p_v_dot_p_f_ext_n = f_jacs
        else:
            p_v_dot_p_f_ext_nm1, p_v_dot_p_f_ext_n = None, None

        # pseudo-acceleration
        d_a = dict()
        d_a["v_dot_nm1"], d_a["v_dot_n"], d_a["a_nm1"], d_a["a_n"] = jax.jacrev(
            self.a_res_func, argnums=(0, 1, 2, 3)
        )(
            q_nm1.v_dot.ravel(),
            q_n.v_dot.ravel(),
            q_nm1.a.ravel(),
            q_n.a.ravel(),
            solve_dofs,
        )

        struct_sizes = (
            len(solve_dofs),
            len(solve_dofs),
            len(solve_dofs),
            len(solve_dofs),
        )

        p_r_n_p_q_nm1 = construct_named_block_jacobian(
            entries=tuple(
                [
                    {k: v[:, solve_dofs] for k, v in jacs.items()}
                    for jacs in (d_varphi, d_v, d_v_dot, d_a)
                ]
            ),
            keys=("varphi_nm1", "v_nm1", "v_dot_nm1", "a_nm1"),
            widths=struct_sizes,
            heights=struct_sizes,
        )

        p_r_n_p_q_n = construct_named_block_jacobian(
            entries=tuple(
                [
                    {k: v[:, solve_dofs] for k, v in jacs.items()}
                    for jacs in (d_varphi, d_v, d_v_dot, d_a)
                ]
            ),
            keys=("varphi_n", "v_n", "v_dot_n", "a_n"),
            widths=struct_sizes,
            heights=struct_sizes,
        )

        return (
            p_r_n_p_q_nm1,
            p_r_n_p_q_n,
            p_v_dot_p_dv,
            p_v_dot_p_f_ext_nm1,
            p_v_dot_p_f_ext_n,
        )

    def j_from_q_x(
        self,
        q_n_mat: Array,
        dv: StructuralDesignVariables,
        dv_full: StructuralDesignVariables,
        objective: StructuralObjectiveFunction,
        i_ts: int,
    ) -> Array:
        r"""
        Obtain the objective as a function of the minimal states and design variables.
        :param q_n_mat: Matrix representation of the minimal states.
        :param dv: Design variables, with unwanted entries replaced with None.
        :param dv_full: Design variables which are defined for all entries.
        :param objective: Objective function.
        :param i_ts: Time step index.
        :return: Objective value.
        """
        full_states = self.minimal_states_to_full_states(
            i_ts=i_ts,
            q=StructureMinimalStates.from_mat(q_n_mat),
            dv=dv,
            dv_full=dv_full,
        )
        return jnp.atleast_1d(objective(full_states, dv, i_ts))

    @jax.jit(static_argnums=(0, 1, 3, 4))
    def p_j(
        self,
        objective: StructuralObjectiveFunction,
        i_ts: int,
        dv: StructuralDesignVariables,
        dv_full: StructuralDesignVariables,
        q_n: StructureMinimalStates,
    ) -> tuple[Array, StructuralDesignVariables]:
        r"""
        Obtains Jacobians of the objective function.
        :param objective: Objective function.
        :param i_ts: Time step index.
        :param dv: Design variables.
        :param dv_full: Design variables which are defined for all entries.
        :param q_n: Current minimal states.
        :return: Jacobian with respect to minimal states and design variables.
        """

        def _j(q_n_mat: Array, dv_: StructuralDesignVariables) -> Array:
            return self.j_from_q_x(
                q_n_mat=q_n_mat, dv=dv_, dv_full=dv_full, objective=objective, i_ts=i_ts
            )

        p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(_j, argnums=(0, 1))(q_n.to_mat(), dv)

        return cast(Array, p_j_n_p_q_n), cast(StructuralDesignVariables, p_j_n_p_x)

    @jax.jit(static_argnums=(0, 6, 7, 8, 9, 11, 12, 13, 14))
    def adjoint_time_loop(
        self,
        rev_i_ts: int,
        d_j_d_x_: StructuralDesignVariables,
        adj_: Array,
        p_r_np1_p_q_n: Array,
        q_n: StructureMinimalStates,
        structure: DynamicStructure,
        objective: StructuralObjectiveFunction,
        dv: StructuralDesignVariables,
        dv_full: StructuralDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
        save_adjoint: bool,
        n_j: int,
    ) -> tuple[StructuralDesignVariables, Array, Array, StructureMinimalStates]:
        r"""
        Function to obtain the grads states at timestep varphi, which is dependent on the grads at timestep varphi+1.
        :param rev_i_ts: Reversed timestep index. JAX loop does not allow for reverse indexing, and so this is.
        explicitly reversed within the function body to obtain i_ts.
        :param d_j_d_x_: Design gradient to accumulate.
        :param adj_: Full grads matrix which is updated inplace, [n_tstep, *j_shape, 5*n_dof].
        :param p_r_np1_p_q_n: Gradient of future step with respect to current state, [5*n_dof, 5*n_dof].
        :param q_n: Current minimal states.
        :param structure: Dynamic structure solution.
        :param objective: Objective function.
        :param dv: Structure design variables.
        :param dv_full: Structure design variables which are defined for all entries.
        :param thrust_t: Thrust time history, {key, [n_tstep]}.
        :param solve_dofs: Tuple of dof index to solve.
        :param approx_grads: Whether to approximate the gradient or not.
        :param save_adjoint: Whether to save the full adjoint time history.
        :param n_j: Number of objective function outputs.
        :return: Updated grads matrix, gradient of current step with respect to previous state and current state.
        """

        i_ts = (
            structure.n_tstep - rev_i_ts - 1
        )  # index for timestep varphi, which decrements

        i_ts_nm1 = jnp.maximum(i_ts - 1, 0)  # index for timestep varphi-1

        # find minimal states for timestep varphi-1
        q_nm1 = structure.get_minimal_states(i_ts_nm1)

        # gradient of objective at current timestep with respect to current minimal states and design variables
        # for i_ts=0, these will not be useful
        p_j_n_p_q_n, p_j_n_p_x = self.p_j(
            objective=objective, i_ts=i_ts, dv=dv, dv_full=dv_full, q_n=q_n
        )

        # find gradients of residual function (state Jacobians only)
        p_r_n_p_q_nm1, p_r_n_p_q_n, p_r_v_dot_n_p_dv, _, _ = (
            self.timestep_residual_jacobians(
                i_ts=i_ts,
                q_n=q_n,
                q_nm1=q_nm1,
                dv=dv,
                solve_dofs=solve_dofs,
                approx_grads=approx_grads,
                f_ext_aero_nm1=None,
                f_ext_aero_n=None,
                thrust_t=thrust_t,
            )
        )

        # solve for adjoint at current timestep
        prev_adjoint = adj_[i_ts + 1, ...] if save_adjoint else adj_
        b: Array = -(p_j_n_p_q_n.reshape(n_j, -1) + prev_adjoint @ p_r_np1_p_q_n).T
        adj_n = jnp.linalg.solve(p_r_n_p_q_n.T, b).T

        if save_adjoint:
            adj_ = adj_.at[i_ts, ...].set(adj_n)

        # accumulate design derivative
        d_j_d_x_ += p_r_v_dot_n_p_dv.premultiply_adj(
            adj_n[:, jnp.array(solve_dofs) + 2 * len(solve_dofs)]
        )

        # add on direct contribution from objective
        d_j_d_x_ += p_j_n_p_x

        jax_print(
            "Adjoint step: {i_ts}",
            i_ts=i_ts,
            verbose_level=VerbosityLevel.NORMAL,
        )

        return d_j_d_x_, adj_ if save_adjoint else adj_n, p_r_n_p_q_nm1, q_nm1

    def dynamic_adjoint(
        self,
        structure: DynamicStructure,
        objective: StructuralObjectiveFunction,
        p_q0_p_x: Optional[StructuralDesignVariables] = None,
        save_adjoint: bool = False,
        approx_grads: bool = False,
        grads_to_compute: StructuralGradsToCompute = StructuralGradsToCompute(
            x0=False,
            k_cs=True,
            m_cs=True,
            m_lumped=False,
            f_ext_follower=False,
            f_ext_dead=False,
        ),
    ) -> tuple[StructuralDesignVariables, Optional[Array]]:
        r"""
        Dynamic structure grads problem. This computes the gradient of the objective of the dynamic response with
        respect to design variables. The objective has structure
        :math:`J = \sum_{i=1}^N \left(j(\mathbf{x}, \mathbf{y}_i)\right)` where :math:`\mathbf{x}` are the design variables
        and :math:`\mathbf{y}` are the structural states at each timestep, which depend on the design variables through
        the dynamic structure equations. The gradient is computed by first solving a backward pass to obtain the grads
        states, and then using these to compute the gradient w.r.t. design variables in a forward pass.
        :param structure: Dynamic structure solution object.
        :param objective: Objective function :math:`j(\mathbf{x}, \mathbf{y}_i)`
        :param p_q0_p_x: Optional Jacobian used to describe the sensitivities of the initial structural degrees of
        freedom to the design variables.
        :param save_adjoint: Whether to save the full adjoint vectors.
        :param approx_grads: If true, some gradient contributions which are assumed to be near-zero are removed to
        decrease computational cost.
        :param grads_to_compute: Design variables with which to compute design gradients for.
        :return: Objective gradient :math:`\frac{dJ}{d\mathbf{x}}` and adjoint states
        """

        dv = self.get_design_variables(
            struct_case=structure,
            thrust_t=structure.thrust,
            grads_to_compute=grads_to_compute,
        )

        dv_full = self.get_design_variables(
            struct_case=structure, thrust_t=structure.thrust, grads_to_compute=None
        )

        struct_states_init = structure.get_full_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(struct_states_init, dv, None))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        # assemble
        solve_dofs: tuple[int, ...] = tuple(
            int(i)
            for i in get_solve_dofs(
                n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs
            )
        )

        dv_grad_init = StructuralDesignVariables(
            x0=jnp.zeros((*j_shape, *self.x0.shape)) if dv.x0 is not None else None,
            orientation_euler=jnp.zeros((*j_shape, 3))
            if dv.orientation_euler is not None
            else None,
            k_cs=jnp.zeros((*j_shape, *self.k_cs.shape))
            if dv.k_cs is not None
            else None,
            m_cs=jnp.zeros((*j_shape, *self.m_cs.shape))
            if dv.m_cs is not None
            else None,
            m_lumped=jnp.zeros((*j_shape, *self.m_lumped.shape))
            if self.use_lumped_mass and dv.m_lumped is not None
            else None,
            f_ext_dead=jnp.zeros((*j_shape, *structure.f_ext_dead.shape))
            if structure.f_ext_dead is not None and dv.f_ext_dead is not None
            else None,
            f_ext_follower=jnp.zeros((*j_shape, *structure.f_ext_follower.shape))
            if structure.f_ext_follower is not None and dv.f_ext_follower is not None
            else None,
            thrust_t={
                k: jnp.zeros((*j_shape, *v.shape)) for k, v in structure.thrust.items()
            }
            if dv.thrust_t is not None
            else None,
            f_shape=(),
        )

        n_adj_dof = 4 * (
            self.n_dof - len(structure.prescribed_dofs)
        )  # number of grads degrees of freedom

        # wrap in a local JIT so structure/aero_dv become closure constants
        @jax.jit
        def adjoint_step(
            rev_i_ts_: int,
            d_j_d_x_: Array,
            adj_: Array,
            p_r_np1_p_q_n: Array,
            q_n: StructureMinimalStates,
        ) -> tuple[Array, Array, Array, StructureMinimalStates]:
            return self.adjoint_time_loop(
                rev_i_ts=rev_i_ts_,
                d_j_d_x_=d_j_d_x_,
                adj_=adj_,
                p_r_np1_p_q_n=p_r_np1_p_q_n,
                q_n=q_n,
                structure=structure,
                objective=objective,
                dv=dv,
                dv_full=dv_full,
                thrust_t=structure.thrust,
                solve_dofs=solve_dofs,
                approx_grads=approx_grads,
                save_adjoint=save_adjoint,
                n_j=n_j,
            )

        # pass through time steps backwards to obtain adjoints
        d_j_d_x, adj, p_r1_p_q0, _ = jax.lax.fori_loop(
            lower=0,
            upper=structure.n_tstep - 1,
            body_fun=lambda i_ts_, args: adjoint_step(i_ts_, *args),
            init_val=(
                dv_grad_init,
                jnp.zeros((structure.n_tstep + 1, n_j, n_adj_dof))
                if save_adjoint
                else jnp.zeros((n_j, n_adj_dof)),
                jnp.zeros((n_adj_dof, n_adj_dof)),
                structure.get_minimal_states(-1),
            ),
        )

        # solve initial timestep adjoint, as there is no r0
        p_j0_p_q0, p_j0_p_x = self.p_j(
            objective=objective,
            i_ts=0,
            dv=dv,
            dv_full=dv_full,
            q_n=structure.get_minimal_states(0),
        )

        adj0 = (
            -p_j0_p_q0.reshape(n_j, -1)
            - (adj[1, ...] if save_adjoint else adj) @ p_r1_p_q0
        )

        # add initial direct sensitivity
        d_j_d_x += p_j0_p_x

        # include initial state sensitivity
        if p_q0_p_x is not None:
            d_j_d_x += p_q0_p_x.premultiply_adj(-adj0)

        # restore original shape of j, and cut off zeros for past-end timestep
        if save_adjoint:
            adj = adj.at[0, ...].set(adj0)
            return d_j_d_x, adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[:-1]
        else:
            return d_j_d_x, None
