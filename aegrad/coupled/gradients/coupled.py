from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aero.gradients.data_structures import AeroDesignVariables
from algebra.array_utils import ArrayList
from coupled import DynamicAeroelastic
from structure import StructuralDesignVariables
from structure.data_structures import OptionalJacobians

from coupled.gradients.data_structures import (
    AeroelasticFullStates,
    AeroelasticDesignVariables,
    AeroelasticDesignGradients, AeroelasticMinimalStates,
)

from coupled.coupled import BaseCoupledAeroelastic
from aero.flowfields import Constant
from algebra.se3 import exp_se3
from structure.gradients.data_structures import StructureFullStates
from structure.utils import get_solve_dofs

if TYPE_CHECKING:
    from coupled.coupled import StaticAeroelastic

type AeroelasticObjectiveFunction = Callable[
    [AeroelasticFullStates, AeroelasticDesignVariables, Optional[int | Array]], Array
]


class CoupledAeroelastic(BaseCoupledAeroelastic):
    def _aeroelastic_states_res_from_dv_varphi(
            self,
            dv: AeroelasticDesignVariables,
            varphi: Array,
            i_ts: int,
            t: Array,
            use_horseshoe: bool,
    ) -> tuple[AeroelasticFullStates, Array]:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        """

        # make a copy of the structure_dv object to prevent modifying the original states
        inner_case = deepcopy(self)

        inner_case.set_design_variables(
            coords=dv.structure.x0,
            k_cs=dv.structure.k_cs,
            m_cs=dv.structure.m_cs,
            m_lumped=dv.structure.m_lumped,
            dt=self.aero.dt,
            flowfield=Constant(
                u_inf=dv.aero.u_inf,
                rho=dv.aero.rho,
                relative_motion=self.aero.flowfield.relative_motion,
            ),  # TODO: generalise
            delta_w=self.aero.delta_w,
            x0_aero=dv.aero.x0_aero,
            remove_checks=True,
        )

        exp_varphi = vmap(exp_se3)(varphi.reshape(-1, 6))  # [n_nodes_, 4, 4]
        hg = jnp.einsum(
            "ijk,ikl->ijl", inner_case.structure.hg0, exp_varphi
        )  # [n_nodes_, 4, 4]

        # evaluate aero forcing and project to beam nodes
        aero_sol = inner_case.aero.solve_static(hg=hg, t=t, horseshoe=use_horseshoe)
        f_ext_aero_global = aero_sol.project_forcing_to_beam(
            i_ts=0, rmat=hg[:, :3, :3], x0_aero=self.aero.x0_b, include_unsteady=False
        )

        d = inner_case.structure.make_d(hg)
        p_d = inner_case.structure.make_p_d(d)
        eps = inner_case.structure.make_eps(d)
        f_int = inner_case.structure.assemble_vector_from_entries(
            inner_case.structure.make_f_int(p_d, eps)
        ).reshape(-1, 6)
        if inner_case.structure.use_gravity:
            m_t = inner_case.structure.make_m_t(jax.lax.stop_gradient(d))
        else:
            m_t = None

        if dv.structure.f_ext_dead is not None:
            f_ext_dead = inner_case.structure.make_f_dead_ext(
                dv.structure.f_ext_dead, hg[:, :3, :3]
            )
        else:
            f_ext_dead = None

        struct_states = StructureFullStates(
            hg=hg,
            eps=eps,
            f_int=f_int,
            v=None,
            v_dot=None,
        )

        f_dead_total = inner_case.structure.make_f_ext_dead_tot(
            f_ext_dead, f_ext_aero_global, i_load_step=None
        )

        f_res = inner_case.structure.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.structure.f_ext_follower,
            f_ext_dead_n=f_dead_total,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        aero_states = aero_sol.get_states(i_ts=i_ts)

        return AeroelasticFullStates(structure=struct_states, aero=aero_states), f_res

    def minimal_states_to_full_states(self, q: AeroelasticMinimalStates) -> AeroelasticFullStates:
        return AeroelasticFullStates(structure=self.structure.minimal_states_to_full_states(q.structure), aero=q.aero)

    @jax.jit(static_argnums=(0, 1, 2, 3))
    def static_adjoint(
            self,
            case: StaticAeroelastic,
            objective: AeroelasticObjectiveFunction,
            optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
                True, True, True, True
            ),
    ) -> AeroelasticDesignGradients:
        r"""
        Computes the static grads of the structure_dv, which is used to compute gradients of the loss with respect to
        the structure_dv's parameters.
        :param case: StaticAeroelastic containing the current state of the aeroelastic system.
        :param objective: Objective function that takes the structure_dv and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :return: Gradient of objective function output with respect to design variables.
        """

        solve_dofs = get_solve_dofs(n_dof=self.structure.n_dof, prescribed_dofs=case.structure.prescribed_dofs)

        if optional_jacobians is not None:
            self.structure.optional_jacobians = optional_jacobians

        dv = self.get_design_variables(case=case)
        states = case.get_full_states()

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(states, dv, None))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.structure.n_x + dv.aero.n_x
        n_u_full = self.structure.n_dof

        varphi = case.structure.varphi

        # gradient of objective w.r.t. minimal states and design variables
        p_f_p_varphi, p_f_p_x = jax.jacrev(
            lambda flat_varphi, dv_: objective(
                self._aeroelastic_states_res_from_dv_varphi(
                    dv_, flat_varphi.reshape(self.structure.n_nodes, 6), t=case.aero.t, i_ts=0,
                    use_horseshoe=case.aero.horseshoe
                )[0],
                dv_, None
            ), argnums=(0, 1)
        )(varphi.ravel(), dv)  # [n_f, n_u_full], [n_f, n_x]

        # gradient of residual w.r.t. minimal states and design variables (used by linear solves)
        p_res_p_varphi, p_res_p_x = jax.jacrev(
            lambda flat_varphi, dv_: self._aeroelastic_states_res_from_dv_varphi(
                dv_, flat_varphi.reshape(self.structure.n_nodes, 6), t=case.aero.t, i_ts=0,
                use_horseshoe=case.aero.horseshoe
            )[1], argnums=(0, 1)
        )(varphi.ravel(), dv)

        adj = (
                jnp.linalg.solve(
                    p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)].T,
                    p_f_p_varphi.reshape(n_f, -1)[:, solve_dofs].T,
                ).T
                @ p_res_p_x.ravel_jacobian(f_size=n_u_full, x_size=n_x)[solve_dofs, :]
        )

        d_f_d_x_dict = dv.from_adjoint(
            f_shape, p_f_p_x.ravel_jacobian(f_size=n_f, x_size=n_x) - adj
        )

        return dv.split_adjoint(d_f_d_x=d_f_d_x_dict, f_shape=f_shape)

    def dynamic_adjoint(self,
                        case: DynamicAeroelastic,
                        objective: AeroelasticObjectiveFunction,
                        free_wake: bool, gamma_dot_relaxation: float = 0.7,
                        p_q0_p_x: Optional[AeroelasticDesignVariables] = None,
                        ) -> tuple[AeroelasticDesignVariables, Array]:
        r"""
        Compute the adjoint of a coupled dynamic aeroelastic system.
        :param case: Dynamic aeroelastic case
        :param objective: Objective function that takes the system full states, design variables and timestep index, and returns an array
        :param free_wake: Whether to use a free wake formulation.
        :param gamma_dot_relaxation: Damping for gamma dot computation. TODO: include this and free_wake in case object
        :param p_q0_p_x: Gradient of initial states with respect to design variables. In practice, this is found from the static solve.
        :return: Gradient of sum of objective across timesteps with respect to design variables.
        """

        solve_dofs = get_solve_dofs(n_dof=self.structure.n_dof, prescribed_dofs=case.structure.prescribed_dofs)

        dv = self.get_design_variables(case=case)

        def timestep_residual(
                i_ts: int,
                t: Array,
                q_nm1: AeroelasticMinimalStates,
                q_n: AeroelasticMinimalStates,
                dv_: AeroelasticDesignVariables,
        ) -> Array:
            inner_case = self.case_from_dv(dv=dv_)

            # compute structural residual
            r_struct: Array = inner_case.structure.timestep_residual(i_ts=i_ts, q_nm1=q_nm1.structure,
                                                                     q_n=q_n.structure,
                                                                     dv_=dv_.structure).ravel()

            # obtain node coordinates and coordinate velocities
            hg_n = inner_case.structure.calculate_hg_from_varphi(varphi=q_n.structure.varphi)
            hg_dot_n = inner_case.structure.make_hg_dot(hg=hg_n, v=q_n.structure.v)

            if q_n.structure.f_ext_aero is None: raise ValueError("f_ext_aero not defined")

            # compute aero residual
            r_aero: Array = inner_case.aero.timestep_residual(hg_n=hg_n, hg_dot_n=hg_dot_n, t_n=t, q_nm1=q_nm1.aero,
                                                              q_n=q_n.aero, free_wake=free_wake, dv=dv_.aero,
                                                              gamma_dot_relaxation=gamma_dot_relaxation,
                                                              f_aero_beam_n=q_n.structure.f_ext_aero)

            return jnp.concatenate((r_struct, r_aero))

            # make copy of structure_dv which has been converted to global coordinates, used to extract dead forces.

        gs = deepcopy(case.structure)
        gs.to_global()

        dv = self.get_design_variables(case=case)

        full_states_init = case.get_full_states(i_ts=0)
        minimal_states_init = case.get_minimal_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(full_states_init, dv, None))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        def p_r_n(
                i_ts: int,
                t: Array,
                q_nm1: AeroelasticMinimalStates,
                q_n: AeroelasticMinimalStates,
                dv_: AeroelasticDesignVariables,
        ) -> tuple[Array, Array, AeroelasticDesignVariables]:
            def inner(q_nm1_vec: Array, q_n_vec: Array, dv__: AeroelasticDesignVariables) -> Array:
                q_nm1_ = AeroelasticMinimalStates.from_vector(vect=q_nm1_vec, n_dof=self.structure.n_dof,
                                                              aero_shapes=minimal_states_init.aero.shapes())

                q_n_ = AeroelasticMinimalStates.from_vector(vect=q_n_vec, n_dof=self.structure.n_dof,
                                                            aero_shapes=minimal_states_init.aero.shapes())

                return timestep_residual(i_ts=i_ts, t=t, q_nm1=q_nm1_, q_n=q_n_, dv_=dv__).ravel()  # [n_adj_dof]

            return jax.jacrev(inner, argnums=(0, 1, 2))(
                q_nm1.ravel(),
                q_n.ravel(),
                dv_,
            )

        def time_loop(
                rev_i_ts: int,
                d_j_d_x_: AeroelasticDesignVariables,
                adj_: Array,
                p_r_np1_p_q_n: Array,
                q_n: AeroelasticMinimalStates,
        ) -> tuple[AeroelasticDesignVariables, Array, Array, AeroelasticMinimalStates]:
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

            i_ts = case.structure.n_tstep - rev_i_ts - 1  # index for timestep varphi, which decrements
            t_n = case.structure.t[i_ts]  # current time

            i_ts_nm1 = jnp.maximum(i_ts - 1, 0)  # index for timestep varphi-1

            # find minimal states for timestep varphi-1
            q_nm1 = case.get_minimal_states(i_ts=i_ts_nm1)

            # gradient of objective at current timestep with respect to current minimal states and design variables
            # for i_ts=0, these will not be useful
            p_j_n_p_q_n: Array
            p_j_n_p_x: AeroelasticDesignVariables
            p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(
                lambda q_vec, dv__: jnp.atleast_1d(
                    objective(
                        self.minimal_states_to_full_states(
                            AeroelasticMinimalStates.from_vector(vect=q_vec, n_dof=self.structure.n_dof,
                                                                 aero_shapes=minimal_states_init.aero.shapes()),
                        ),
                        dv__,
                        i_ts,
                    )
                ),
                argnums=(0, 1),
            )(q_n.ravel(), dv)

            # find gradients of residual function
            p_r_n_p_q_nm1, p_r_n_p_q_n, p_r_n_p_dv = p_r_n(
                i_ts=i_ts, t=t_n, q_n=q_n, q_nm1=q_nm1, dv_=dv
            )

            # solve for adjoint at current timestep
            b: Array = -(
                    p_j_n_p_q_n.reshape(n_j, -1)
                    + adj_[i_ts + 1, ...] @ p_r_np1_p_q_n
            ).T
            adj_ = adj_.at[i_ts, ...].set(
                jnp.linalg.solve(
                    p_r_n_p_q_n.T,
                    b,
                ).T
            )

            jax.debug.print(
                "Solved grads for timestep {i_ts}",
                i_ts=i_ts,
            )

            # accumulate design derivative with adj_.T @ p_s_p_x
            # do not add anything for timestep 0 as the Jacobians refer to ts = -1
            # handle this with another routine, and use zero adjoint states
            d_j_d_x_ += p_r_n_p_dv.premult_adj(adj_[i_ts, ...])

            # add on direct contribution from objective
            d_j_d_x_ += p_j_n_p_x

            return d_j_d_x_, adj_, p_r_n_p_q_nm1, q_nm1

        dv_grad_init = AeroelasticDesignVariables(
            structure_dv=StructuralDesignVariables(x0=jnp.zeros((*j_shape, *self.structure.x0.shape)),
                                                   k_cs=jnp.zeros((*j_shape, *self.structure.k_cs.shape)),
                                                   m_cs=jnp.zeros((*j_shape, *self.structure.m_cs.shape)),
                                                   m_lumped=jnp.zeros((*j_shape, *self.structure.m_lumped.shape))
                                                   if self.structure.use_lumped_mass
                                                   else None,
                                                   f_ext_dead=jnp.zeros((*j_shape, *gs.f_ext_dead.shape))
                                                   if gs.f_ext_dead is not None
                                                   else None,
                                                   f_ext_follower=jnp.zeros((*j_shape, *gs.f_ext_follower.shape))
                                                   if gs.f_ext_follower is not None
                                                   else None),
            aero_dv=AeroDesignVariables(
                x0_aero=ArrayList([jnp.zeros((*j_shape, *arr.shape)) for arr in self.aero.x0_b]),
                u_inf=jnp.zeros((*j_shape, 3)),
                rho=jnp.zeros((*j_shape,)))
        )

        n_adj_dof = minimal_states_init.n_states

        # pass through time steps backwards to obtain adjoints
        d_j_d_x, adj, p_r1_p_q0, _ = jax.lax.fori_loop(
            0,
            case.structure.n_tstep - 1,
            lambda i_ts, args: time_loop(i_ts, *args),
            init_val=(
                dv_grad_init,
                jnp.zeros((case.structure.n_tstep + 1, n_j, n_adj_dof)),
                jnp.zeros((n_adj_dof, n_adj_dof)),
                case.get_minimal_states(i_ts=-1),
            ),
        )

        # solve initial timestep adjoint, as there is no r0
        p_j0_p_q0: Array
        p_j0_p_x: StructuralDesignVariables
        p_j0_p_q0, p_j0_p_x = jax.jacrev(
            lambda q_vec, dv__: jnp.atleast_1d(
                objective(
                    self.minimal_states_to_full_states(
                        AeroelasticMinimalStates.from_vector(vect=q_vec, n_dof=self.structure.n_dof,
                                                             aero_shapes=minimal_states_init.aero.shapes())
                    ),
                    dv__,
                    0,
                )
            ),
            argnums=(0, 1),
        )(case.get_minimal_states(0).ravel(), dv)

        adj = adj.at[0, ...].set(-p_j0_p_q0.reshape(n_j, -1) - adj[1, ...] @ p_r1_p_q0)

        # add initial direct sensitivity
        d_j_d_x += p_j0_p_x

        # include initial state sensitivity
        if p_q0_p_x is not None:
            d_j_d_x += p_q0_p_x.premult_adj(-adj[0, ...])

        # restore original shape of j, and cut off zeros for past-end timestep
        adj = adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[:-1]

        return d_j_d_x, adj
