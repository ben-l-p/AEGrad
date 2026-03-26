from __future__ import annotations
from copy import copy, deepcopy
from typing import Optional, Callable

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.structure.beam import BaseBeamStructure
from aegrad.structure import OptionalJacobians
from aegrad.structure.data_structures import (
    StaticStructure,
)
from aegrad.structure.gradients.data_structures import (
    StructureFullStates,
    StructuralDesignVariables,
)
from algebra.se3 import exp_se3, t_se3, t_inv_se3, hg_to_ha_hat
from structure import DynamicStructure
from structure.gradients.data_structures import UnsteadyStructureMinimalStates

type StructuralObjectiveFunction = Callable[
    [StructureFullStates, StructuralDesignVariables], Array
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
        f_int = inner_case.assemble_vector_from_entries(
            inner_case.make_f_int(p_d, eps)
        ).reshape(-1, 6)
        if inner_case.use_gravity:
            m_t = inner_case.make_m_t(d)
        else:
            m_t = None

        ss = StructureFullStates(
            hg=hg,
            eps=eps,
            f_int=f_int,
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
        Computes the static adjoint of the structure_dv, which is used to compute gradients of the loss with respect to
        the structure_dv's parameters.
        :param structure: StaticStructure containing the current state of the structure_dv.
        :param objective: Objective function that takes the structure_dv and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :return: Gradient of objective function output with respect to design variables.
        """

        solve_dofs = jnp.setdiff1d(
            jnp.arange(self.n_dof),
            structure.prescribed_dofs,
            size=self.n_dof - structure.prescribed_dofs.size,
        )
        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # base parameters
        varphi = self.calculate_varphi_from_hg(structure.hg)  # [n_nodes, 6]
        t_n = vmap(t_se3, 0, 0)(varphi.reshape(-1, 6))  # [n_nodes_, 6, 6]
        d = structure.d
        eps = structure.eps
        p_d = self.make_p_d(d)
        m_t = self.make_m_t(d) if self.use_gravity else None

        # make copy of structure_dv which has been converted to global coordinates.
        gs = copy(structure)
        gs.to_global()

        # make design variables for current state of structure_dv
        dv = StructuralDesignVariables(
            x0=self.x0,
            k_cs=self.k_cs,
            m_cs=self.m_cs if self.use_m_cs else None,
            m_lumped=self.m_lumped if self.use_lumped_mass else None,
            f_ext_follower=structure.f_ext_follower,
            f_ext_dead=gs.f_ext_dead,
        )

        struct_states = StructureFullStates(
            hg=structure.hg,
            eps=eps,
            f_int=structure.f_int,
            v_dot=None,
            v=None,
        )

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)
        n_u_full = self.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n = jax.jacrev(
            lambda varphi_: objective(
                self._structural_states_res_from_dv_varphi(dv, varphi_)[0], dv
            )
        )(varphi).reshape(n_f, n_u_full)[:, solve_dofs]  # [n_f, n_u]

        # gradient of objective w.r.t. design variables
        p_f_p_x = (jax.jacrev if n_f < n_x else jax.jacfwd)(
            lambda dv_: objective(
                self._structural_states_res_from_dv_varphi(dv_, varphi)[0], dv_
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

        # gradient of residual w.r.t. design variables
        p_res_p_x = (jax.jacrev if n_u < n_x else jax.jacfwd)(
            lambda dv_: self._structural_states_res_from_dv_varphi(dv_, varphi)[1]
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

    def dynamic_adjoint(
        self,
        structure: DynamicStructure,
        objective: StructuralObjectiveFunction,
        optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
            True, True, True, True
        ),
    ) -> StructuralDesignVariables:
        r"""
        Dynamic structure adjoint problem. This computes the gradient of the objective of the dynamic response with
        respect to design variables. The objective has structure
        :math:`J = \sum_{i=1}^N \left(j(\mathbf{x}, \mathbf{y}_i)\right)` where :math:`\mathbf{x}` are the design variables
        and :math:`\mathbf{y}` are the structural states at each timestep, which depend on the design variables through
        the dynamic structure equations. The adjoint is computed by first solving a backward pass to obtain the adjoint
        states, and then using these to compute the gradient w.r.t. design variables in a forward pass.
        :param structure: Dynamic structure solution object.
        :param objective: Objective function :math:`j(\mathbf{x}, \mathbf{y}_i)`
        :param optional_jacobians: Optional Jacobians to use for solution
        :return: Objective gradient :math:`\frac{dJ}{d\mathbf{x}}`
        """

        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # make copy of structure_dv which has been converted to global coordinates, used to extract dead forces.
        gs = copy(structure)
        gs.to_global()

        # TODO: this comes with assumption of constant force over time

        dv = StructuralDesignVariables(
            x0=self.x0,
            k_cs=self.k_cs,
            m_cs=self.m_cs if self.use_m_cs else None,
            m_lumped=self.m_lumped if self.use_lumped_mass else None,
            f_ext_follower=structure.f_ext_follower,
            f_ext_dead=gs.f_ext_dead,
        )

        alpha_f = self.time_integrator.alpha_f
        alpha_m = self.time_integrator.alpha_m
        beta = self.time_integrator.beta
        gamma = self.time_integrator.gamma
        dt = self.time_integrator.dt

        struct_states_init = StructureFullStates(
            v=structure.v[0, ...],
            v_dot=structure.v_dot[0, ...],
            hg=structure.hg[0, ...],
            eps=structure.eps[0, ...],
            f_int=structure.f_int[0, ...],
        )

        j_properties = jax.eval_shape(lambda: objective(struct_states_init, dv))
        j_shape = j_properties.shape
        n_j = j_properties.size

        # assemble
        solve_dofs = jnp.setdiff1d(
            jnp.arange(self.n_dof),
            structure.prescribed_dofs,
            size=self.n_dof - len(structure.prescribed_dofs),
        )

        n_free_dof = self.n_dof - len(structure.prescribed_dofs)

        def p_s_n(
            i_ts: int,
            q_nm1: UnsteadyStructureMinimalStates,
            q_n: UnsteadyStructureMinimalStates,
        ) -> tuple[Array, Array]:
            r"""
            Obtain the Jacobians :math:`\frac{\partial \mathbf{s}_{n}}{\partial \mathbf{q}_{n-1}}` and
            :math:`\frac{\partial \mathbf{s}_{n}}{\partial \mathbf{q}_{n}}` for system minimal states
            :math:`\mathbf{q}` with stepping function :math:`\mathbf{q}_{n} =
            \mathbf{s}_{n}(\mathbf{q}_{n-1}, \mathbf{q}_{n})`.
            :param i_ts: Index for which to obtain s
            :param q_nm1: Minimal states at timestep n-1.
            :param q_n: Minimal states at timestep n.
            :return: Jacobians :math:`\frac{\partial \mathbf{s}_{n}}{\partial \mathbf{q}_{n-1}}` and :math:`\frac{
            \partial \mathbf{s}_{n}}{\partial \mathbf{q}_{n}}`
            """
            q_alpha = self.time_integrator.calculate_q_alpha(q_n=q_nm1, q_np1=q_n)
            hg_alpha = self.calculate_hg_from_varphi(q_alpha.varphi)
            d_alpha = self.make_d(hg_alpha)
            p_d_alpha = self.make_p_d(d=d_alpha)
            eps_alpha = self.make_eps(d=d_alpha)

            f_ext_dead_alpha = (
                self.time_integrator.calculate_f_alpha(
                    f_n=gs.f_ext_dead[i_ts, ...], f_np1=gs.f_ext_dead[i_ts + 1, ...]
                )
                if gs.f_ext_dead is not None
                else None
            )

            m_alpha_entries = self.make_m_t(d=d_alpha)
            m_alpha = self.assemble_matrix_from_entries(m_alpha_entries)
            k_t = self._make_k_t_full(
                d=d_alpha,
                p_d=p_d_alpha,
                eps=eps_alpha,
                rmat=hg_alpha[:, :3, :3],
                m_t=m_alpha if self.use_gravity else None,
                f_ext_dead=f_ext_dead_alpha,
            )
            d_dot_alpha = self._make_d_dot(p_d=p_d_alpha, v=q_alpha.v)
            c_t = self.assemble_matrix_from_entries(
                self._make_c_t(d=d_alpha, v=q_alpha.v, d_dot=d_dot_alpha)[1]
            )

            t_varphi_nm1 = vmap(t_se3, 0, 0)(q_nm1.varphi)  # [n_nodes, 6, 6]
            t_inv_varphi_n = vmap(t_inv_se3, 0, 0)(q_n.varphi)  # [n_nodes, 6, 6]
            t_phi_n = vmap(t_se3, 0, 0)(q_n.phi)

            ad_exp_neg_phi_n = vmap(lambda phi_: hg_to_ha_hat(exp_se3(-phi_)), 0, 0)(
                q_n.phi
            )  # [n_nodes, 6, 6]

            m_alpha_lu = jax.scipy.linalg.lu_factor(m_alpha)

            m_inv_k = jax.scipy.linalg.lu_solve(m_alpha_lu, k_t)
            m_inv_c = jax.scipy.linalg.lu_solve(m_alpha_lu, c_t)

            a1 = jax.scipy.linalg.block_diag(
                *jnp.einsum(
                    "ijk,ikl,ilm->ijm", t_inv_varphi_n, ad_exp_neg_phi_n, t_varphi_nm1
                )
            )
            a2 = -alpha_f * m_inv_k @ jax.scipy.linalg.block_diag(*t_varphi_nm1)
            a3 = -alpha_f * m_inv_c
            a4 = (1.0 - alpha_f) / (1.0 - alpha_m) * a2
            a5 = (1.0 - alpha_f) / (1.0 - alpha_m) * a3
            a6 = jax.scipy.linalg.block_diag(
                *jnp.einsum("ijk,ikl->ijl", t_inv_varphi_n, t_phi_n)
            )
            a7 = (1.0 - alpha_f) / alpha_f * a2
            a8 = (1.0 - alpha_f) / alpha_f * a3
            a9 = (1.0 - alpha_f) / (1.0 - alpha_m) * a7
            a10 = (1.0 - alpha_f) / (1.0 - alpha_m) * a8

            p_s_n_p_q_nm1 = jnp.zeros((5 * self.n_dof, 5 * self.n_dof))
            p_s_n_p_q_n = jnp.zeros((5 * self.n_dof, 5 * self.n_dof))

            dof_slices = []
            ix = []
            for i_slice in range(5):
                dof_slices.append(
                    slice(i_slice * self.n_dof, (i_slice + 1) * self.n_dof)
                )

                ix.append(solve_dofs + i_slice * self.n_dof)
            ix = jnp.concatenate(ix, axis=0)  # [5 * n_free_dof]

            eye = jnp.eye(self.n_dof)

            # assemble matrix w.r.t previous state
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[0], dof_slices[2]].set(dt * eye)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[0], dof_slices[4]].set(
                (0.5 - beta) * dt * dt * eye
            )
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[1], dof_slices[1]].set(a1)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[2], dof_slices[2]].set(eye)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[2], dof_slices[4]].set(
                (1.0 - gamma) * dt * eye
            )
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[3], dof_slices[1]].set(a2)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[3], dof_slices[2]].set(a3)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[4], dof_slices[1]].set(a4)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[4], dof_slices[2]].set(a5)
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[4], dof_slices[3]].set(
                alpha_f / (1.0 - alpha_m) * eye
            )
            p_s_n_p_q_nm1 = p_s_n_p_q_nm1.at[dof_slices[4], dof_slices[4]].set(
                alpha_m / (alpha_m - 1.0) * eye
            )

            # assemble matrix w.r.t current state
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[0], dof_slices[4]].set(
                beta * dt * dt * eye
            )
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[1], dof_slices[0]].set(a6)
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[2], dof_slices[4]].set(
                gamma * dt * eye
            )
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[3], dof_slices[1]].set(a7)
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[3], dof_slices[2]].set(a8)
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[4], dof_slices[1]].set(a9)
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[4], dof_slices[2]].set(a10)
            p_s_n_p_q_n = p_s_n_p_q_n.at[dof_slices[4], dof_slices[3]].set(
                (1.0 - alpha_f) / (1.0 - alpha_m) * eye
            )

            return p_s_n_p_q_nm1[jnp.ix_(ix, ix)], p_s_n_p_q_n[jnp.ix_(ix, ix)]

        def minimal_states_to_full_states(
            q_n: UnsteadyStructureMinimalStates,
        ) -> StructureFullStates:
            r"""
            Obtain the full structural states :math:`\mathbf{y}` from minimal states :math:`\mathbf{y}`.
            :param q_n: Minimal states
            :return: Full states
            """
            hg = self.calculate_hg_from_varphi(q_n.varphi)
            d = self.make_d(hg=hg)
            eps = self.make_eps(d=d)
            p_d = self.make_p_d(d=d)
            f_int = self.make_f_int(eps=eps, p_d=p_d)
            return StructureFullStates(
                v=q_n.v, v_dot=q_n.v_dot, eps=eps, hg=hg, f_int=f_int
            )

        def solve_adjoint_states() -> Array:
            r"""
            Solve the system in reverse to obtain the time series of adjoint variables.
            :return: Adjoint variables, [n_tstep, *j_shape, 5*n_dof]
            """

            def adjoint_state_loop(
                rev_i_ts: int,
                adj_: Array,
                p_s_np1_p_q_n: Array,
                q_n: UnsteadyStructureMinimalStates,
            ) -> tuple[Array, Array, UnsteadyStructureMinimalStates]:
                r"""
                Function to obtain the adjoint states at timestep n, which is dependent on the adjoint at timestep n+1.
                :param rev_i_ts: Reversed timestep index. JAX fori_loop does not allow for reverse indexing, and so this is explicitly reversed witin the function body to obtain i_ts.
                :param adj_: Full adjoint matrix which is updated inplace, [n_tstep, *j_shape, 5*n_dof]
                :param p_s_np1_p_q_n: Gradient of future step with respect to current state, [5*n_dof, 5*n_dof]
                :param q_n: Current minimal states
                :return: Updated adjoint matrix, gradient of current step with respect to previous state and current state.
                """

                i_ts = (
                    structure.n_tstep - rev_i_ts - 1
                )  # time step index, which decrements

                # minimal states at timestep n-1

                varphi_nm1 = self.calculate_varphi_from_hg(structure.hg[i_ts - 1, ...])
                phi_nm1 = self.calculate_phi_from_hg(
                    structure.hg[
                        jnp.maximum(i_ts - 2, 0), ...
                    ],  # included to catch the case for i_ts=0 having no previous timestep
                    structure.hg[i_ts - 1, ...],
                )

                q_nm1 = UnsteadyStructureMinimalStates(
                    v=structure.v[i_ts - 1, ...],
                    v_dot=structure.v_dot[i_ts - 1, ...],
                    a=structure.a[i_ts - 1, ...],
                    phi=phi_nm1,
                    varphi=varphi_nm1,
                )

                # gradient of objective at current timestep with respect to current minimal states
                p_j_n_p_q_n = jax.jacrev(
                    lambda q_mat: objective(
                        minimal_states_to_full_states(
                            UnsteadyStructureMinimalStates.from_mat(q_mat)
                        ),
                        dv,
                    )
                )(q_nm1.to_mat()).reshape(n_j, -1)  # [n_j, 5*n_dof]

                # find gradients of stepping function
                # the first array is required for solving the next timestep
                p_s_n_p_q_nm1, p_s_n_p_q_n = p_s_n(i_ts=i_ts, q_n=q_n, q_nm1=q_nm1)

                # solve for adjoint at current timestep
                a = jnp.eye(p_s_np1_p_q_n.shape[0]) - p_s_n_p_q_n
                b = p_j_n_p_q_n + adj_[i_ts + 1, ...] @ p_s_np1_p_q_n
                adj_ = adj_.at[i_ts, ...].set(jnp.linalg.solve(a.T, b.T).T)

                jax.debug.print(
                    "Solved adjoint for timestep {i_ts}/{n_tstep}",
                    i_ts=i_ts,
                    n_tstep=structure.n_tstep,
                )

                return adj_, p_s_n_p_q_nm1, q_nm1

            # obtain minimal states for the final timestep, used as initialisation for backpass
            varphi_end = self.calculate_varphi_from_hg(structure.hg[-1, ...])
            phi_end = self.calculate_phi_from_hg(
                structure.hg[-2, ...], structure.hg[-1, ...]
            )

            q_end = UnsteadyStructureMinimalStates(
                v=structure.v[-1, ...],
                v_dot=structure.v_dot[-1, ...],
                a=structure.a[-1, ...],
                phi=phi_end,
                varphi=varphi_end,
            )
            n_adj_dof = 5 * (
                self.n_dof - len(structure.prescribed_dofs)
            )  # number of adjoint degrees of freedom

            # pass through time steps backwards to obtain adjoints
            adj, _, _ = jax.lax.fori_loop(
                0,
                structure.n_tstep,
                lambda i_ts, args: adjoint_state_loop(i_ts, *args),
                init_val=(
                    jnp.zeros((structure.n_tstep + 1, n_j, n_adj_dof)),
                    jnp.zeros((n_adj_dof, n_adj_dof)),
                    q_end,
                ),
            )

            return adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[
                :-1
            ]  # restore original shape of j, and cut off zeros for past-end timestep

        def obtain_d_j_n_d_dv() -> StructuralDesignVariables:
            r"""
            Obtain the gradient of the objective with respect to the design variables, evaluated as the sum across all
            timestep states (as the gradient w.r.t. design variables may be dependent on the time-dependent states).
            :return: StructuralDesignVariables object which includes the summed gradients.
            """
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

            def inner_dv_grad_loop(
                i_ts: int, dv_grad: StructuralDesignVariables
            ) -> StructuralDesignVariables:
                r"""
                Inner loop which finds the gradient contribution from a single timestep.
                :param i_ts: Timestep index
                :param dv_grad: Gradient of the objective with respect to the design variables
                :return: Updated gradient
                """
                ss = StructureFullStates(
                    v=structure.v[i_ts, ...],
                    v_dot=structure.v_dot[i_ts, ...],
                    hg=structure.hg[i_ts, ...],
                    eps=structure.eps[i_ts, ...],
                    f_int=structure.f_int[i_ts, ...],
                )

                jac = jax.jacrev(lambda dv_: objective(ss, dv_))(dv)

                dv_grad.x0 += jac.x0
                dv_grad.k_cs += jac.k_cs
                dv_grad.m_cs += jac.m_cs
                if self.use_lumped_mass:
                    dv_grad.m_lumped += jac.m_lumped
                if structure.f_ext_dead is not None:
                    dv_grad.f_ext_dead += jac.f_ext_dead
                if structure.f_ext_follower is not None:
                    dv_grad.f_ext_follower += jac.f_ext_follower

                return dv_grad

            return jax.lax.fori_loop(
                0, structure.n_tstep, inner_dv_grad_loop, dv_grad_init
            )

        def s(
            i_ts: int,
            q_nm1: UnsteadyStructureMinimalStates,
            q_n: UnsteadyStructureMinimalStates,
            dv_: StructuralDesignVariables,
        ) -> UnsteadyStructureMinimalStates:
            r"""
            Function which steps the structural problem forward from timestep varphi to varphi+1 implicitly.
            :param i_ts: Timestep index
            :param q_nm1: Mimimal structural states at timestep n-1
            :param q_n: Minimal structural states at timestep n
            :param dv_: Structural design variables
            :return: Structural states at timestep n
            """

            # state updates obtained from time integrator without knowledge of structural problem
            phi_n = (
                dt * q_nm1.v + (0.5 - beta) * dt * dt * q_nm1.a + beta * dt * dt * q_n.a
            )
            varphi_n = self.calculate_varphi_from_phi(
                varphi_n=q_nm1.varphi, phi_np1=q_n.phi
            )

            v_n = q_nm1.v + (1.0 - gamma) * dt * q_nm1.a + gamma * dt * q_n.a

            a_n = (
                1.0
                / (1.0 - alpha_m)
                * (
                    (1.0 - alpha_f) * q_n.v_dot
                    + alpha_f * q_nm1.v_dot
                    - alpha_m * q_nm1.a
                )
            )

            # updates to v_dot, which are obtained from relation to other states through structural problem

            inner_case = self.case_from_dv(
                dv=dv_
            )  # allows for gradients w.r.t. design variables

            # solve problem between timesteps, as should be done for the used time integrator method
            q_alpha = inner_case.time_integrator.calculate_q_alpha(q_n=q_nm1, q_np1=q_n)
            hg_alpha = inner_case.calculate_hg_from_varphi(q_alpha.varphi)

            f_ext_dead_alpha = (
                inner_case.time_integrator.calculate_f_alpha(
                    f_n=dv_.f_ext_dead[i_ts, ...], f_np1=dv_.f_ext_dead[i_ts + 1, ...]
                )
                if dv_.f_ext_dead is not None
                else None
            )
            f_ext_follower_alpha = (
                inner_case.time_integrator.calculate_f_alpha(
                    f_n=structure.f_ext_follower[i_ts, ...],
                    f_np1=structure.f_ext_follower[i_ts + 1, ...],
                )
                if structure.f_ext_follower is not None
                else None
            )

            d_, eps_, f_dead_res, _, f_grav_res, f_int, f_iner, f_res = (
                inner_case._resolve_forces(
                    hg=hg_alpha,
                    dynamic=True,
                    f_ext_follower=f_ext_follower_alpha,
                    f_ext_dead=f_ext_dead_alpha,
                    f_ext_aero=None,
                    v=q_alpha.v,
                    v_dot=q_alpha.v_dot,
                )
            )

            m_alpha = inner_case.assemble_matrix_from_entries(inner_case.make_m_t(d=d_))
            if self.use_lumped_mass:
                m_alpha += jax.scipy.linalg.block_diag(*inner_case.m_lumped)

            f_res_non_iner = f_int + f_iner
            if self.use_gravity:
                f_res_non_iner += f_grav_res
            if f_dead_res is not None:
                f_res_non_iner += f_dead_res
            if f_ext_follower_alpha is not None:
                f_res_non_iner += f_ext_follower_alpha
            v_dot_alpha = -jnp.linalg.solve(m_alpha, f_res_non_iner.ravel()).reshape(
                -1, 6
            )

            # find v_dot_np1 from its alpha value
            v_dot_n = (v_dot_alpha - alpha_f * q_nm1.v_dot) / (1.0 - alpha_f)

            return UnsteadyStructureMinimalStates(
                phi=phi_n, varphi=varphi_n, v=v_n, a=a_n, v_dot=v_dot_n
            )

        adj_states = solve_adjoint_states()
        d_j_n_d_dv = obtain_d_j_n_d_dv()

        varphi_0 = self.calculate_varphi_from_hg(structure.hg[0, ...])
        q_0 = UnsteadyStructureMinimalStates(
            v=structure.v[0, ...],
            v_dot=structure.v_dot[0, ...],
            a=structure.a[0, ...],
            phi=jnp.zeros_like(varphi_0),
            varphi=varphi_0,
        )

        def inner_loop(
            i_ts: int,
            d_dv: StructuralDesignVariables,
            q_n: UnsteadyStructureMinimalStates,
        ) -> tuple[StructuralDesignVariables, UnsteadyStructureMinimalStates]:
            r"""
            Inner loop for the forward final pass of the adjoint to accumulate the result.
            :param i_ts: Timestep index
            :param d_dv: Gradient :math`\frac{dJ}{d\mathbf{x}}` accumulated from previous timesteps
            :param q_n: Current minimal state
            :return: Update gradient, and next state
            """

            # adj[i_ts] is the adjoint of state q_{i_ts}, the output of the
            # transition from q_{i_ts-1}. Shape: [n_j, n_adj_dof].
            adj_ts = adj_states[i_ts, ...]

            varphi_np1 = self.calculate_varphi_from_hg(structure.hg[i_ts, ...])
            phi_np1 = self.calculate_phi_from_hg(
                structure.hg[i_ts - 1, ...], structure.hg[i_ts, ...]
            )
            q_np1 = UnsteadyStructureMinimalStates(
                v=structure.v[i_ts, ...],
                v_dot=structure.v_dot[i_ts, ...],
                a=structure.a[i_ts, ...],
                phi=phi_np1,
                varphi=varphi_np1,
            )

            # Partial derivative of the state transition s_{i_ts-1} w.r.t. dv_,
            # evaluated at the stored (q_nm1, q_nm1). Captured via VJP.
            _, vjp_fn = jax.vjp(lambda dv_: s(i_ts - 1, q_n, q_np1, dv_), dv)

            # Accumulate adj_ts^T @ (∂s_{i_ts-1}/∂dv_) for each objective.
            # adj_ts: [n_j, 5*n_free_dof] — scatter free DOFs back into full
            # state space, reshape to UnsteadyStructureMinimalStates, then vmap
            # over n_j. vjp_fn returns a 1-tuple so we unpack with (grad_dv,).
            all_free_idx = jnp.concatenate(
                [solve_dofs + k * self.n_dof for k in range(5)]
            )

            def apply_vjp_single(g_vec):
                g_full = jnp.zeros(5 * self.n_dof).at[all_free_idx].set(g_vec)
                cotangent = UnsteadyStructureMinimalStates.from_mat(
                    g_full.reshape(5, self.n_nodes, 6)
                )
                (grad_dv,) = vjp_fn(cotangent)
                return grad_dv

            p_s_n_p_dv = jax.vmap(apply_vjp_single)(adj_ts.reshape(n_j, 5 * n_free_dof))

            d_dv += p_s_n_p_dv
            return d_dv, q_np1

        dv_adj_init = StructuralDesignVariables(
            x0=jnp.zeros((n_j, *self.x0.shape)),
            k_cs=jnp.zeros((n_j, *self.k_cs.shape)),
            m_cs=jnp.zeros((n_j, *self.m_cs.shape)) if dv.m_cs is not None else None,
            m_lumped=jnp.zeros((n_j, *self.m_lumped.shape))
            if dv.m_lumped is not None
            else None,
            f_ext_dead=jnp.zeros((n_j, *dv.f_ext_dead.shape))
            if dv.f_ext_dead is not None
            else None,
            f_ext_follower=jnp.zeros((n_j, *dv.f_ext_follower.shape))
            if dv.f_ext_follower is not None
            else None,
        )

        dv_adj, _ = jax.lax.fori_loop(
            1,
            structure.n_tstep,
            lambda i_ts, args: inner_loop(i_ts, *args),
            init_val=(dv_adj_init, q_0),
        )

        # Combine direct gradient (d_j_n_d_dv) with adjoint contribution (dv_adj).
        return StructuralDesignVariables(
            x0=d_j_n_d_dv.x0 + dv_adj.x0.reshape(*j_shape, *self.x0.shape),
            k_cs=d_j_n_d_dv.k_cs + dv_adj.k_cs.reshape(*j_shape, *self.k_cs.shape),
            m_cs=d_j_n_d_dv.m_cs + dv_adj.m_cs.reshape(*j_shape, *self.m_cs.shape)
            if dv.m_cs is not None
            else d_j_n_d_dv.m_cs,
            m_lumped=d_j_n_d_dv.m_lumped
            + dv_adj.m_lumped.reshape(*j_shape, *self.m_lumped.shape)
            if dv.m_lumped is not None
            else d_j_n_d_dv.m_lumped,
            f_ext_dead=d_j_n_d_dv.f_ext_dead
            + dv_adj.f_ext_dead.reshape(*j_shape, *gs.f_ext_dead.shape)
            if dv.f_ext_dead is not None
            else d_j_n_d_dv.f_ext_dead,
            f_ext_follower=d_j_n_d_dv.f_ext_follower,
        )
