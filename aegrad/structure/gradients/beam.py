from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable

import jax
from jax import numpy as jnp
from jax import Array, vmap

from utils.print_utils import jax_print, VerbosityLevel
from structure.beam import BaseBeamStructure
from structure import OptionalJacobians
from structure.data_structures import (
    StaticStructure,
)
from structure.gradients.data_structures import (
    StructureFullStates,
    StructuralDesignVariables,
)
from algebra.se3 import t_se3, exp_se3, log_se3
from structure import DynamicStructure
from structure.data_structures import StructureMinimalStates
from structure.utils import get_solve_dofs, transform_nodal_vect

type StructuralObjectiveFunction = Callable[
    [StructureFullStates, StructuralDesignVariables, Optional[int | Array]], Array
]


class BeamStructure(BaseBeamStructure):
    def case_from_dv(self, dv: StructuralDesignVariables) -> BeamStructure:
        r"""
        Obtain a structural object as a function of design variables, allowing it to have defined gradients w.r.t. design variables.
        :param dv: Design variables.
        :return: Beam structure object with the same functionality as self.
        """
        inner_case = deepcopy(self)
        inner_case.set_design_variables(
            coords=dv.x0,
            k_cs=dv.k_cs,
            m_cs=dv.m_cs,
            m_lumped=dv.m_lumped,
            remove_checks=True,
        )

        return inner_case

    def minimal_states_to_full_states(self, q: StructureMinimalStates,
                                      dv: Optional[StructuralDesignVariables] = None) -> StructureFullStates:
        struct = self.case_from_dv(dv) if dv is not None else self
        hg = struct.calculate_hg_from_varphi(q.varphi)
        d = struct.make_d(hg=hg)
        eps = struct.make_eps(d=d)
        f_elem = struct.make_f_elem(eps=eps)
        return StructureFullStates(v=q.v, v_dot=q.v_dot, eps=eps, hg=hg, f_elem=f_elem)

    def _structural_states_res_from_dv_varphi(
            self,
            dv: StructuralDesignVariables,
            varphi: Array,
    ) -> tuple[StructureFullStates, Array]:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        :param dv: Design variables.
        :param varphi: Twist coordinates which map from the reference configuration to the current as
        :math:`\mathbf{H} = \mathbf{H}_0 \mathrm{exp} (\varphi)`
        :return: Structural states and forcing residual.
        """

        inner_case = self.case_from_dv(dv=dv)

        hg = inner_case.calculate_hg_from_varphi(varphi=varphi)  # [n_nodes_, 4, 4]
        d = inner_case.make_d(hg)
        p_d = inner_case.make_p_d(d)
        eps = inner_case.make_eps(d)
        f_elem = jnp.einsum('ijk,ik->ij', inner_case.k_cs, eps)

        if inner_case.use_gravity:
            m_t = inner_case.make_m_t(d)
        else:
            m_t = None

        ss = StructureFullStates(
            hg=hg,
            eps=eps,
            f_elem=f_elem,
            v=None,
            v_dot=None,
        )

        f_res = inner_case.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.f_ext_follower,
            f_ext_dead_n=dv.f_ext_dead,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        return ss, f_res

    def static_adjoint(
            self,
            structure: StaticStructure,
            objective: StructuralObjectiveFunction,
            optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
                True, True, True, True
            ),
    ) -> StructuralDesignVariables:
        r"""
        Computes the static grads of the structure_dv, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param structure: StaticStructure containing the current state of the structure_dv.
        :param objective: Objective function that takes the structure_dv and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :return: Gradient of objective function output with respect to design variables.
        """

        solve_dofs = get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs)

        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # base parameters
        t_n = vmap(t_se3, 0, 0)(structure.varphi)  # [n_nodes_, 6, 6]
        d = structure.d
        eps = structure.eps
        p_d = self.make_p_d(d)
        m_t = self.make_m_t(d) if self.use_gravity else None
        f_elem = self.make_f_elem(eps=eps)

        # Recover original global dead force: structure.f_ext_dead = R @ f_global (local),
        # so f_global = R^T @ structure.f_ext_dead
        rmat_t = jnp.transpose(structure.hg[:, :3, :3], (0, 2, 1))
        f_ext_dead_global = (
            transform_nodal_vect(structure.f_ext_dead, rmat_t)
            if structure.f_ext_dead is not None else None
        )

        # make design variables for current state of structure_dv
        dv = StructuralDesignVariables(
            x0=self.x0,
            k_cs=self.k_cs,
            m_cs=self.m_cs if self.use_m_cs else None,
            m_lumped=self.m_lumped if self.use_lumped_mass else None,
            f_ext_follower=structure.f_ext_follower,
            f_ext_dead=f_ext_dead_global,
        )

        struct_states = StructureFullStates(
            hg=structure.hg,
            eps=eps,
            f_elem=f_elem,
            v_dot=None,
            v=None,
        )

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv, None))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)
        n_u_full = self.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n = jax.jacrev(
            lambda varphi_: objective(
                self._structural_states_res_from_dv_varphi(dv, varphi_)[0], dv, None
            )
        )(structure.varphi).reshape(n_f, n_u_full)[:, solve_dofs]  # [n_f, n_u]

        # gradient of objective w.r.t. design variables
        p_f_p_x = (jax.jacrev if n_f < n_x else jax.jacfwd)(
            lambda dv_: objective(
                self._structural_states_res_from_dv_varphi(dv_, structure.varphi)[0],
                dv_,
                None,
            )
        )(dv).ravel_jacobian(n_f, n_x)  # [n_f, n_x]

        # gradient of residual w.r.t. minimal states
        # h and varphi have different tangent spaces, p_h_p_n = t(varphi)
        d_res_d_h = -self._make_k_t_full(
            d, p_d, eps, structure.f_ext_dead, structure.hg[:, :3, :3], m_t
        ).reshape(n_u_full, -1, 6)  # [n_u_full, n_nodes_, 6]
        d_res_d_n = jnp.einsum("ijk,jkl->ijl", d_res_d_h, t_n).reshape(
            n_u_full, n_u_full
        )[jnp.ix_(solve_dofs, solve_dofs)]

        # gradient of residual w.r.t. minimal states via JAX (exact)
        # d_res_d_n = jax.jacfwd(
        #     lambda varphi_: self._structural_states_res_from_dv_varphi(dv, varphi_)[1]
        # )(structure.varphi).reshape(n_u_full, n_u_full)[jnp.ix_(solve_dofs, solve_dofs)]

        # gradient of residual w.r.t. design variables
        p_res_p_x = (jax.jacrev if n_u < n_x else jax.jacfwd)(
            lambda dv_: self._structural_states_res_from_dv_varphi(
                dv_, structure.varphi
            )[1]
        )(dv).ravel_jacobian(n_u_full, n_x)[solve_dofs, :]  # [n_u, n_x]

        if n_f > n_x:
            # forward mode
            d_n_d_x = jnp.linalg.solve(d_res_d_n, p_res_p_x).reshape(
                self.n_nodes, 6, n_x
            )  # [n_u, n_x]
            rhs = jnp.einsum("ij,ijk->k", p_f_p_n, d_n_d_x)  # [n_f, n_x]
        else:
            # reverse mode
            d_f_d_res = jnp.linalg.solve(d_res_d_n.T, p_f_p_n.T).T  # [n_f, n_u]
            rhs = d_f_d_res @ p_res_p_x  # [n_f, n_x]

        return StructuralDesignVariables(**dv.from_adjoint(f_shape, p_f_p_x - rhs))

    def timestep_residual(
            self,
            i_ts: int,
            q_nm1: StructureMinimalStates,
            q_n: StructureMinimalStates,
            dv_: StructuralDesignVariables,
            solve_dofs: Optional[Array] = None,
    ) -> Array:
        r"""
        Function which finds the residual of the structural problem forward from timestep n-1 to timestep n.
        :param i_ts: Timestep index n
        :param q_nm1: Minimal structural states at timestep n-1.
        :param q_n: Minimal structural states at timestep n.
        :param dv_: Structural design variables.
        :param solve_dofs: Array of dofs to solve for each timestep n.
        :return: Residual for step.
        """

        # time integrator parameters
        dt = self.time_integrator.dt
        beta = self.time_integrator.beta
        gamma = self.time_integrator.gamma
        alpha_f = self.time_integrator.alpha_f
        alpha_m = self.time_integrator.alpha_m

        # state updates obtained from time integrator without knowledge of structural problem
        phi_n = (
                dt * q_nm1.v + (0.5 - beta) * dt * dt * q_nm1.a + beta * dt * dt * q_n.a
        )

        varphi_res = vmap(
            lambda vp_n, vp_nm1, phi: log_se3(
                exp_se3(-vp_n) @ exp_se3(vp_nm1) @ exp_se3(phi)
            ),
            0,
            0,
            0,
        )(q_n.varphi, q_nm1.varphi, phi_n).ravel()[solve_dofs]

        v_res = (q_nm1.v + (1.0 - gamma) * dt * q_nm1.a + gamma * dt * q_n.a - q_n.v).ravel()[solve_dofs]

        a_res = ((
                         (1.0 - alpha_f) * q_n.v_dot + alpha_f * q_nm1.v_dot - alpha_m * q_nm1.a
                 ) / (1.0 - alpha_m) - q_n.a).ravel()[solve_dofs]

        # updates to v_dot, which are obtained from relation to other states through structural problem

        inner_case = self.case_from_dv(
            dv=dv_
        )  # allows for gradients w.r.t. design variables

        # solve problem between timesteps, as should be done for the used time integrator method
        phi_alpha, q_alpha = inner_case.time_integrator.calculate_q_alpha(
            q_nm1=q_nm1, q_n=q_n, phi_n=phi_n
        )
        hg_alpha = inner_case.calculate_hg_from_varphi(q_alpha.varphi)

        i_ts_nm1 = jnp.maximum(i_ts - 1, 0)

        f_ext_dead_alpha = (
            inner_case.time_integrator.calculate_f_alpha(
                f_nm1=dv_.f_ext_dead[i_ts_nm1, ...], f_n=dv_.f_ext_dead[i_ts, ...]
            )
            if dv_.f_ext_dead is not None
            else None
        )
        f_ext_follower_alpha = (
            inner_case.time_integrator.calculate_f_alpha(
                f_nm1=dv_.f_ext_follower[i_ts_nm1, ...],
                f_n=dv_.f_ext_follower[i_ts, ...],
            )
            if dv_.f_ext_follower is not None
            else None
        )

        if q_n.f_ext_aero is not None and q_nm1.f_ext_aero is not None:
            f_aero_alpha = inner_case.time_integrator.calculate_f_alpha(f_nm1=q_nm1.f_ext_aero, f_n=q_n.f_ext_aero)
        else:
            f_aero_alpha = None

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
            v=q_alpha.v,
            v_dot=q_alpha.v_dot,
            stop_gradients=False
        )

        # use stop gradient to prevent effective stiffness contribution
        m_alpha = inner_case.assemble_matrix_from_entries(
            inner_case.make_m_t(d=d_alpha)
        )
        if inner_case.use_lumped_mass:
            m_alpha = inner_case.add_lumped_contributions_to_arr(arr=m_alpha, lumped_arr=inner_case.m_lumped)

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
        if solve_dofs is None:
            solve_dofs_: Array | ellipsis = ...
        else:
            solve_dofs_: Array | ellipsis = solve_dofs
            m_alpha = m_alpha[jnp.ix_(solve_dofs, solve_dofs)]

        # find the v_dot residual to satisfy the structural problem
        v_dot_res = (f_res_non_iner.ravel()[solve_dofs_] / (1.0 - alpha_f) - m_alpha @ (
                alpha_f / (1.0 - alpha_f) * q_nm1.v_dot.ravel()[solve_dofs_] + q_n.v_dot.ravel()[solve_dofs_]))

        return jnp.stack(
            (varphi_res, v_res, v_dot_res, a_res), axis=0
        ).ravel()  # [4*n_free_dof]

    # @jax.jit(static_argnums=(0, 1, 2, 3))
    def dynamic_adjoint(
            self,
            structure: DynamicStructure,
            objective: StructuralObjectiveFunction,
            p_q0_p_x: Optional[StructuralDesignVariables] = None,
            optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
                True, True, True, True
            ),
    ) -> tuple[StructuralDesignVariables, Array]:
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
        :param optional_jacobians: Optional Jacobians to use for solution
        :return: Objective gradient :math:`\frac{dJ}{d\mathbf{x}}` and adjoint states
        """

        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # make copy of structure_dv which has been converted to global coordinates, used to extract dead forces.
        gs = deepcopy(structure)
        gs.to_global()

        dv = self.get_design_variables(struct_case=structure)

        struct_states_init = structure.get_full_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(struct_states_init, dv, None))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        # assemble
        solve_dofs = get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs)

        free_state_ix = jnp.concatenate([solve_dofs + i * self.n_dof for i in range(4)])

        def p_r_n(
                i_ts: int,
                q_nm1: StructureMinimalStates,
                q_n: StructureMinimalStates,
                dv_: StructuralDesignVariables,
        ) -> tuple[Array, Array, StructuralDesignVariables]:
            def inner(
                    q_nm1_free: Array, q_n_free: Array, dv__: StructuralDesignVariables
            ) -> Array:
                q_nm1_struct = StructureMinimalStates.from_mat(
                    q_nm1.to_mat()
                    .ravel()
                    .at[free_state_ix]
                    .set(q_nm1_free)
                    .reshape(4, -1, 6)
                )
                q_n_struct = StructureMinimalStates.from_mat(
                    q_n.to_mat()
                    .ravel()
                    .at[free_state_ix]
                    .set(q_n_free)
                    .reshape(4, -1, 6)
                )
                return self.timestep_residual(
                    i_ts=i_ts, q_nm1=q_nm1_struct, q_n=q_n_struct, dv_=dv__, solve_dofs=solve_dofs
                )

            return jax.jacrev(inner, argnums=(0, 1, 2))(
                q_nm1.to_mat().ravel()[free_state_ix],
                q_n.to_mat().ravel()[free_state_ix],
                dv_,
            )

        def minimal_states_to_full_states(
                q_n: StructureMinimalStates,
        ) -> StructureFullStates:
            r"""
            Obtain the full structural states :math:`\mathbf{y}` from minimal states :math:`\mathbf{y}`.
            :param q_n: Minimal states
            :return: Full states
            """
            hg = self.calculate_hg_from_varphi(q_n.varphi)
            d = self.make_d(hg=hg)
            eps = self.make_eps(d=d)
            f_elem = self.make_f_elem(eps=eps)
            return StructureFullStates(
                v=q_n.v, v_dot=q_n.v_dot, eps=eps, hg=hg, f_elem=f_elem
            )

        def time_loop(
                rev_i_ts: int,
                d_j_d_x_: StructuralDesignVariables,
                adj_: Array,
                p_r_np1_p_q_n: Array,
                q_n: StructureMinimalStates,
        ) -> tuple[StructuralDesignVariables, Array, Array, StructureMinimalStates]:
            r"""
            Function to obtain the grads states at timestep varphi, which is dependent on the grads at timestep varphi+1.
            :param rev_i_ts: Reversed timestep index. JAX loop does not allow for reverse indexing, and so this is.
            explicitly reversed within the function body to obtain i_ts.
            :param d_j_d_x_: Design gradient to accumulate.
            :param adj_: Full grads matrix which is updated inplace, [n_tstep, *j_shape, 5*n_dof].
            :param p_r_np1_p_q_n: Gradient of future step with respect to current state, [5*n_dof, 5*n_dof].
            :param q_n: Current minimal states.
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
            p_j_n_p_q_n: Array
            p_j_n_p_x: StructuralDesignVariables
            p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(
                lambda q_mat, dv__: jnp.atleast_1d(
                    objective(
                        minimal_states_to_full_states(
                            StructureMinimalStates.from_mat(q_mat)
                        ),
                        dv__,
                        i_ts,
                    )
                ),
                argnums=(0, 1),
            )(q_n.to_mat(), dv)

            # find gradients of residual function
            p_r_n_p_q_nm1, p_r_n_p_q_n, p_r_n_p_dv = p_r_n(
                i_ts=i_ts, q_n=q_n, q_nm1=q_nm1, dv_=dv
            )

            # solve for adjoint at current timestep
            b: Array = -(
                    p_j_n_p_q_n.reshape(n_j, -1)[:, free_state_ix]
                    + adj_[i_ts + 1, ...] @ p_r_np1_p_q_n
            ).T
            adj_ = adj_.at[i_ts, ...].set(
                jnp.linalg.solve(
                    p_r_n_p_q_n.T,
                    b,
                ).T
            )

            jax_print("Adjoint for timestep {i_ts} solved", i_ts=i_ts, verbose_level=VerbosityLevel.NORMAL)

            # accumulate design derivative with adj_.T @ p_s_p_x
            # do not add anything for timestep 0 as the Jacobians refer to ts = -1
            # handle this with another routine, and use zero adjoint states
            d_j_d_x_ += p_r_n_p_dv.premult_adj(adj_[i_ts, ...])

            # add on direct contribution from objective
            d_j_d_x_ += p_j_n_p_x

            return d_j_d_x_, adj_, p_r_n_p_q_nm1, q_nm1

        dv_grad_init = StructuralDesignVariables(
            x0=jnp.zeros((*j_shape, *self.x0.shape)),
            k_cs=jnp.zeros((*j_shape, *self.k_cs.shape)),
            m_cs=jnp.zeros((*j_shape, *self.m_cs.shape)),
            m_lumped=jnp.zeros((*j_shape, *self.m_lumped.shape))
            if self.use_lumped_mass
            else None,
            f_ext_dead=jnp.zeros((*j_shape, *gs.f_ext_dead.shape))
            if gs.f_ext_dead is not None
            else None,
            f_ext_follower=jnp.zeros((*j_shape, *gs.f_ext_follower.shape))
            if gs.f_ext_follower is not None
            else None,
        )

        n_adj_dof = 4 * (
                self.n_dof - len(structure.prescribed_dofs)
        )  # number of grads degrees of freedom

        # pass through time steps backwards to obtain adjoints
        d_j_d_x, adj, p_r1_p_q0, _ = jax.lax.fori_loop(
            0,
            structure.n_tstep - 1,
            lambda i_ts, args: time_loop(i_ts, *args),
            init_val=(
                dv_grad_init,
                jnp.zeros((structure.n_tstep + 1, n_j, n_adj_dof)),
                jnp.zeros((n_adj_dof, n_adj_dof)),
                structure.get_minimal_states(-1),
            ),
        )

        # solve initial timestep adjoint, as there is no r0
        p_j0_p_q0: Array
        p_j0_p_x: StructuralDesignVariables
        p_j0_p_q0, p_j0_p_x = jax.jacrev(
            lambda q_mat, dv__: jnp.atleast_1d(
                objective(
                    minimal_states_to_full_states(
                        StructureMinimalStates.from_mat(q_mat)
                    ),
                    dv__,
                    0,
                )
            ),
            argnums=(0, 1),
        )(structure.get_minimal_states(0).to_mat(), dv)

        adj = adj.at[0, ...].set(
            -p_j0_p_q0.reshape(n_j, -1)[:, free_state_ix] - adj[1, ...] @ p_r1_p_q0
        )

        # add initial direct sensitivity
        d_j_d_x += p_j0_p_x

        # include initial state sensitivity
        if p_q0_p_x is not None:
            d_j_d_x += p_q0_p_x.premult_adj(-adj[0, ...])

        # restore original shape of j, and cut off zeros for past-end timestep
        adj = adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[:-1]

        return d_j_d_x, adj
