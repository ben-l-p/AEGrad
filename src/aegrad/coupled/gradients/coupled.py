from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.aero.gradients.data_structures import AeroDesignVariables, AeroStates
from aegrad.aero.utils import project_forcing_to_beam
from aegrad.algebra.array_utils import ArrayList
from aegrad.coupled import DynamicAeroelastic
from aegrad.utils.print_utils import jax_print, VerbosityLevel
from aegrad.structure import StructuralDesignVariables
from aegrad.structure.data_structures import OptionalJacobians, StructureMinimalStates

from aegrad.coupled.data_structures import (
    AeroelasticFullStates,
    AeroelasticMinimalStates,
    AeroelasticDesignVariables,
)

from aegrad.coupled.coupled import BaseCoupledAeroelastic
from aegrad.algebra.se3 import exp_se3
from aegrad.structure.gradients.data_structures import StructureFullStates
from aegrad.structure.utils import get_solve_dofs, transform_nodal_vect

if TYPE_CHECKING:
    from aegrad.coupled.coupled import StaticAeroelastic

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

        # make a copy of the structure object to prevent modifying the original states
        inner_case = deepcopy(self)

        inner_case.set_design_variables(
            coords=dv.structure.x0
            if dv.structure.x0 is not None
            else self.structure.x0,
            k_cs=dv.structure.k_cs
            if dv.structure.k_cs is not None
            else self.structure.k_cs,
            m_cs=dv.structure.m_cs
            if dv.structure.m_cs is not None
            else self.structure.m_cs,
            m_lumped=dv.structure.m_lumped
            if dv.structure.m_lumped is not None
            else None,
            dt=self.aero.dt,
            flowfield=self.aero.flowfield.from_design_variables(
                design_variables=dv.aero.flowfield
            ),
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
        f_elem = inner_case.structure.make_f_elem(eps=eps)

        if inner_case.structure.use_gravity:
            m_t = inner_case.structure.make_m_t(d)
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
            f_elem=f_elem,
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

    def minimal_states_to_full_states(
        self,
        q: AeroelasticMinimalStates,
        dv: Optional[AeroelasticDesignVariables] = None,
    ) -> AeroelasticFullStates:
        return AeroelasticFullStates(
            structure=self.structure.minimal_states_to_full_states(
                q.structure, dv=dv.structure if dv is not None else None
            ),
            aero=q.aero,
        )

    @jax.jit(static_argnums=(0, 1, 2, 3, 4))
    def static_adjoint(
        self,
        case: StaticAeroelastic,
        objective: AeroelasticObjectiveFunction,
        optional_jacobians: Optional[OptionalJacobians] = OptionalJacobians(
            True, True, True, True
        ),
        forward_adjoint: bool = False,
    ) -> tuple[AeroelasticDesignVariables, Array]:
        r"""
        Computes the static grads of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param case: StaticAeroelastic containing the current state of the aeroelastic system.
        :param objective: Objective function that takes the structure and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :param forward_adjoint: If True, will use the forward adjoint. If false, will use the reverse adjoint.
        :return: Gradient of objective function output with respect to design variables.
        """

        solve_dofs = jnp.array(
            get_solve_dofs(
                n_dof=self.structure.n_dof,
                prescribed_dofs=case.structure.prescribed_dofs,
            )
        )

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

        if case.aero.static_horseshoe is None:
            raise ValueError("static_horseshoe not defined")
        static_horseshoe: bool = case.aero.static_horseshoe

        # gradient of objective w.r.t. minimal states and design variables
        p_f_p_varphi, p_f_p_x = jax.jacrev(
            lambda flat_varphi, dv_: objective(
                self._aeroelastic_states_res_from_dv_varphi(
                    dv_,
                    flat_varphi.reshape(self.structure.n_nodes, 6),
                    t=case.aero.t,
                    i_ts=0,
                    use_horseshoe=static_horseshoe,
                )[0],
                dv_,
                None,
            ),
            argnums=(0, 1),
        )(varphi.ravel(), dv)  # [n_f, n_u_full], [n_f, n_x]

        # gradient of residual w.r.t. minimal states and design variables (used by linear solves)
        p_res_p_varphi, p_res_p_x = jax.jacrev(
            lambda flat_varphi, dv_: self._aeroelastic_states_res_from_dv_varphi(
                dv_,
                flat_varphi.reshape(self.structure.n_nodes, 6),
                t=case.aero.t,
                i_ts=0,
                use_horseshoe=static_horseshoe,
            )[1],
            argnums=(0, 1),
        )(varphi.ravel(), dv)

        if forward_adjoint:
            # forward mode — restrict to free DOFs to avoid singularity from prescribed DOF constraints
            adj = jnp.linalg.solve(
                p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)],
                p_res_p_x.ravel_jacobian(f_size=n_u_full, x_size=n_x)[solve_dofs, :],
            )

            d_f_d_x_dict = dv.from_adjoint(
                f_shape,
                p_f_p_x.ravel_jacobian(f_size=n_f, x_size=n_x)
                - p_f_p_varphi.reshape(n_f, -1)[:, solve_dofs] @ adj,
            )
        else:
            # reverse_mode
            adj = jnp.linalg.solve(
                p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)].T,
                p_f_p_varphi.reshape(n_f, -1)[:, solve_dofs].T,
            ).T

            d_f_d_x_dict = dv.from_adjoint(
                f_shape,
                p_f_p_x.ravel_jacobian(f_size=n_f, x_size=n_x)
                - adj
                @ p_res_p_x.ravel_jacobian(f_size=n_u_full, x_size=n_x)[solve_dofs, :],
            )

        return dv.split_adjoint(d_f_d_x=d_f_d_x_dict, f_shape=f_shape), adj

    def compute_p_q0_p_x(
        self,
        case: StaticAeroelastic,
        p_varphi_p_x: Array,
        solve_dofs: Array,
        horseshoe: bool = False,
    ) -> AeroelasticDesignVariables:
        r"""
        Obtain the gradient of the initial minimal states for a dynamic aeroelastic system with respect to the design
        variables.
        :param case: StaticAeroelastic solution
        :param p_varphi_p_x: Jacobian of structural twists with respect to the design variables. This can be obtained
        from the static adjoint solver by using the forward mode, which results in this being the adjoint state.
        :param solve_dofs: Array of degree of freedom index which are solved for
        :param horseshoe: Use static_horseshoe wake.
        :return: Gradients of AeroelasticMinimalStates with respect to the design variables.
        """
        dv = self.get_design_variables(case=case)
        varphi = case.structure.varphi

        def minimal_states_from_varphi(
            varphi_: Array, dv_: AeroelasticDesignVariables
        ) -> Array:
            inner_case = self.case_from_dv(dv=dv_)

            # solve aero problem
            hg = inner_case.structure.calculate_hg_from_varphi(varphi=varphi_)
            c, nc, gamma_b, gamma_w, _, zeta_b, zeta_w, _, f_steady, _ = (
                inner_case.aero.base_solve(
                    q_nm1=None,
                    t_n=case.aero.t,
                    hg_n=hg,
                    hg_dot_n=None,
                    static=True,
                    horseshoe=horseshoe,
                    cs_ang_n=dv_.aero.cs_ang_t,
                    cs_vel_n=dv_.aero.cs_vel_t,
                )
            )

            f_aero_beam_global = project_forcing_to_beam(
                f_total=f_steady,
                rmat=hg[:, :3, :3],
                dof_mapping=inner_case.aero.dof_mapping,
                x0_aero=inner_case.aero.x0_b,
            )

            f_aero_beam_local = transform_nodal_vect(
                vect=f_aero_beam_global, rmat=jnp.transpose(hg[:, :3, :3], (0, 2, 1))
            )

            q_aero = AeroStates(
                gamma_b=gamma_b,
                gamma_w=gamma_w,
                zeta_w=zeta_w,
                gamma_b_dot=ArrayList.zeros_like(gamma_b),
            )

            # assume initial velocities and accelerations are zero
            q_structure = StructureMinimalStates(
                varphi=varphi_,
                v=jnp.zeros_like(varphi_),
                v_dot=jnp.zeros_like(varphi_),
                a=jnp.zeros_like(varphi_),
                f_ext_aero=f_aero_beam_local,
            )

            return AeroelasticMinimalStates(structure=q_structure, aero=q_aero).ravel()

        p_q0_p_varphi, p_q0_p_x = jax.jacrev(
            minimal_states_from_varphi, argnums=(0, 1)
        )(varphi, dv)

        n_q0 = p_q0_p_varphi.shape[0]
        indirect_mat = (
            p_q0_p_varphi.reshape(n_q0, -1)[:, solve_dofs] @ p_varphi_p_x
        )  # (n_q0, n_x)
        indirect_dict = dv.from_adjoint((n_q0,), indirect_mat)
        indirect_term = AeroelasticDesignVariables(
            structure_dv=StructuralDesignVariables(
                **{k: indirect_dict[k] for k in dv.structure.get_vars()}, f_shape=()
            ),
            aero_dv=AeroDesignVariables(
                **{k: indirect_dict[k] for k in dv.aero.get_vars()}, f_shape=()
            ),
        )
        p_q0_p_x += indirect_term
        return p_q0_p_x

    def timestep_residual_jacobians(
        self,
        i_ts: int,
        t: Array,
        q_nm1: AeroelasticMinimalStates,
        q_n: AeroelasticMinimalStates,
        dv_: AeroelasticDesignVariables,
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> tuple[Array, Array, StructuralDesignVariables, AeroelasticDesignVariables]:

        if q_n.structure.f_ext_aero is None or q_nm1.structure.f_ext_aero is None:
            raise ValueError("Missing aerodynamic forcing states")

        (
            p_aero_res_p_q_aero_nm1,
            p_aero_res_p_q_aero_n,
            p_aero_res_d_dv,
            p_aero_res_p_q_struct_nm1,
            p_aero_res_p_q_struct_n,
        ) = self.aero.timestep_residual_jacobians(
            i_ts=i_ts,
            varphi_nm1=q_nm1.structure.varphi,
            varphi_n=q_n.structure.varphi,
            v_n=q_n.structure.v,
            t_n=t,
            q_n=q_n.aero,
            q_nm1=q_nm1.aero,
            dv=dv_,
            f_aero_beam_n=q_n.structure.f_ext_aero,
            struct_obj=self.structure,
            approx_grads=approx_grads,
            use_unsteady=self.aero.include_unsteady_force,
            solve_dofs=solve_dofs,
        )

        n_aero_dof, n_struct_dof = p_aero_res_p_q_struct_nm1.shape

        (
            p_struct_res_p_q_struct_nm1,
            p_struct_res_p_q_struct_n,
            p_v_dot_res_p_struct_dv,
            p_v_dot_res_p_f_ext_nm1,
            p_v_dot_res_p_f_ext_n,
        ) = self.structure.timestep_residual_jacobians(
            i_ts=i_ts,
            q_nm1=q_nm1.structure,
            q_n=q_n.structure,
            f_ext_aero_n=q_n.structure.f_ext_aero,
            f_ext_aero_nm1=q_nm1.structure.f_ext_aero,
            dv=dv_.structure,
            solve_dofs=solve_dofs,
            approx_grads=approx_grads,
        )

        # reduce aero-to-struct Jacobian columns to solve_dofs
        n_solve = len(solve_dofs)
        n_struct_res = p_struct_res_p_q_struct_nm1.shape[0]
        solve_dofs_arr = jnp.array(solve_dofs)
        struct_col_ix = jnp.concatenate(
            [solve_dofs_arr + i * self.structure.n_dof for i in range(4)]
        )
        p_aero_res_p_q_struct_nm1 = p_aero_res_p_q_struct_nm1[:, struct_col_ix]
        p_aero_res_p_q_struct_n = p_aero_res_p_q_struct_n[:, struct_col_ix]

        # create struct-to-aero cross-coupling (only v_dot residual depends on f_ext_aero)
        # f_aero block is n_solve-wide (last n_solve cols of aero state), so slice Jacobian to solve_dofs cols
        p_struct_res_p_q_aero_nm1 = jnp.zeros((n_struct_res, n_aero_dof))
        p_struct_res_p_q_aero_nm1 = p_struct_res_p_q_aero_nm1.at[
            jnp.arange(n_solve) + 2 * n_solve, -n_solve:
        ].set(p_v_dot_res_p_f_ext_nm1[:, solve_dofs_arr])

        p_struct_res_p_q_aero_n = jnp.zeros((n_struct_res, n_aero_dof))
        p_struct_res_p_q_aero_n = p_struct_res_p_q_aero_n.at[
            jnp.arange(n_solve) + 2 * n_solve, -n_solve:
        ].set(p_v_dot_res_p_f_ext_n[:, solve_dofs_arr])

        p_res_p_q_nm1 = jnp.block(
            [
                [p_struct_res_p_q_struct_nm1, p_struct_res_p_q_aero_nm1],
                [p_aero_res_p_q_struct_nm1, p_aero_res_p_q_aero_nm1],
            ]
        )

        p_res_p_q_n = jnp.block(
            [
                [p_struct_res_p_q_struct_n, p_struct_res_p_q_aero_n],
                [p_aero_res_p_q_struct_n, p_aero_res_p_q_aero_n],
            ]
        )

        return p_res_p_q_nm1, p_res_p_q_n, p_v_dot_res_p_struct_dv, p_aero_res_d_dv

    def dynamic_adjoint(
        self,
        case: DynamicAeroelastic,
        objective: AeroelasticObjectiveFunction,
        p_varphi_p_x: Optional[Array] = None,
        save_adjoint: bool = False,
        approx_grads: bool = True,
    ) -> tuple[AeroelasticDesignVariables, Optional[Array]]:
        r"""
        Compute the adjoint of a coupled dynamic aeroelastic system.
        :param case: Dynamic aeroelastic case
        :param objective: Objective function that takes the system full states, design variables and timestep index, and returns an array
        :param p_varphi_p_x: Gradient of initial twists with respect to design variables. In practice, this is found from the static solve.
        :param save_adjoint: Whether to save the adjoint of the dynamic aeroelastic system.
        :param approx_grads: Whether to use gradient approximation or not.
        :return: Gradient of sum of objective across timesteps with respect to design variables.
        """

        # make copies to prevent contaminating input object with tracer
        case = deepcopy(case)
        p_varphi_p_x = deepcopy(p_varphi_p_x)

        solve_dofs: tuple[int, ...] = get_solve_dofs(
            n_dof=self.structure.n_dof,
            prescribed_dofs=case.structure.prescribed_dofs,
        )

        solve_dofs_arr: Array = jnp.array(solve_dofs)

        n_tstep = case.structure.n_tstep

        dv = self.get_design_variables(case=case)

        if case.aero.static_horseshoe is None:
            raise ValueError("static_horseshoe not defined")
        static_horseshoe: bool = case.aero.static_horseshoe

        full_states_init = case.get_full_states(i_ts=0)
        minimal_states_init = case.get_minimal_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(full_states_init, dv, None))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        n_solve = len(solve_dofs)

        j_eval = jnp.array(
            [
                objective(case.get_full_states(i_ts=i_ts), dv, i_ts)
                for i_ts in range(n_tstep)
            ]
        ).reshape(n_tstep, n_j)

        # the last time step where the objective was nonzero. This can be used to prevent redundant computations for
        # computing sensitivities which only refer to a small number of time steps by not running the adjoint problem
        # after the last time step which has a contribution.
        last_active_i_ts = int(jnp.flatnonzero(jnp.any(j_eval, axis=-1))[-1])

        def time_loop(
            rev_i_ts_: int,
            d_j_d_x_: AeroelasticDesignVariables,
            adj_: Array,
            p_r_np1_p_q_n: Array,
            q_n: AeroelasticMinimalStates,
        ) -> tuple[AeroelasticDesignVariables, Array, Array, AeroelasticMinimalStates]:
            r"""
            Function to obtain the grads states at timestep varphi, which is dependent on the grads at timestep varphi+1.
            :param rev_i_ts_: Reversed timestep index. JAX loop does not allow for reverse indexing, and so this is.
            explicitly reversed within the function body to obtain i_ts.
            :param d_j_d_x_: Design gradient to accumulate.
            :param adj_: Full grads matrix which is updated inplace, [n_tstep, *j_shape, n_adj_dof].
            :param p_r_np1_p_q_n: Gradient of future step with respect to current state, [n_adj_dof, n_adj_dof].
            :param q_n: Current minimal states.
            :return: Updated grads matrix, gradient of current step with respect to previous state and current state.
            """

            i_ts = (
                n_tstep - rev_i_ts_ - 1
            )  # index for timestep varphi, which decrements
            t_n = case.structure.t[i_ts]  # current time

            i_ts_nm1 = jnp.maximum(i_ts - 1, 0)  # index for timestep varphi-1

            # find minimal states for timestep varphi-1
            q_nm1 = case.get_minimal_states(i_ts=i_ts_nm1)

            # gradient of objective at current timestep with respect to current minimal states and design variables
            # for i_ts=0, these will not be useful
            p_j_n_p_q_n: Array
            p_j_n_p_x: AeroelasticDesignVariables
            p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(
                lambda q_free, dv__: jnp.atleast_1d(
                    objective(
                        self.minimal_states_to_full_states(
                            AeroelasticMinimalStates.from_vector(
                                vect=q_n.ravel().at[free_state_ix].set(q_free),
                                n_dof=self.structure.n_dof,
                                aero_shapes=minimal_states_init.aero.shapes(),
                            ),
                            dv=dv__,
                        ),
                        dv__,
                        i_ts,
                    )
                ),
                argnums=(0, 1),
            )(q_n.ravel()[free_state_ix], dv)

            # find gradients of residual function using hand-assembled block Jacobians
            p_res_p_q_nm1, p_res_p_q_n, p_v_dot_res_p_struct_dv, p_aero_res_d_dv = (
                self.timestep_residual_jacobians(
                    i_ts=i_ts,
                    t=t_n,
                    q_nm1=q_nm1,
                    q_n=q_n,
                    dv_=dv,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                )
            )

            # solve for adjoint at current timestep
            b: Array = -(
                p_j_n_p_q_n.reshape(n_j, -1)
                + (adj_[i_ts + 1, ...] if save_adjoint else adj_) @ p_r_np1_p_q_n
            ).T

            adj_n = jnp.linalg.solve(
                p_res_p_q_n.T,
                b,
            ).T

            if save_adjoint:
                adj_ = adj_.at[i_ts, ...].set(adj_n)

            jax_print(
                "Solved adjoint for timestep {i_ts}",
                i_ts=i_ts,
                verbose_level=VerbosityLevel.NORMAL,
            )

            # accumulate design derivative: adj^T @ dR/dx
            d_j_d_x_ += p_aero_res_d_dv.premultiply_adj(adj_n[:, 4 * n_solve :])
            d_j_d_x_.structure += p_v_dot_res_p_struct_dv.premultiply_adj(
                adj_n[:, 2 * n_solve : 3 * n_solve]
            )

            # add on direct contribution from objective
            d_j_d_x_ += p_j_n_p_x

            return d_j_d_x_, adj_ if save_adjoint else adj_n, p_res_p_q_nm1, q_nm1

        dv_grad_init = AeroelasticDesignVariables(
            structure_dv=StructuralDesignVariables(
                x0=jnp.zeros((*j_shape, *self.structure.x0.shape)),
                k_cs=jnp.zeros((*j_shape, *self.structure.k_cs.shape)),
                m_cs=jnp.zeros((*j_shape, *self.structure.m_cs.shape)),
                m_lumped=jnp.zeros((*j_shape, *self.structure.m_lumped.shape))
                if self.structure.use_lumped_mass
                else None,
                f_ext_dead=jnp.zeros((*j_shape, *case.structure.f_ext_dead.shape))
                if case.structure.f_ext_dead is not None
                else None,
                f_ext_follower=jnp.zeros(
                    (*j_shape, *case.structure.f_ext_follower.shape)
                )
                if case.structure.f_ext_follower is not None
                else None,
                f_shape=j_shape,
            ),
            aero_dv=AeroDesignVariables(
                x0_aero=ArrayList(
                    [jnp.zeros((*j_shape, *arr.shape)) for arr in self.aero.x0_b]
                ),
                flowfield={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in self.aero.flowfield.to_design_variables().items()
                },
                cs_ang_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in dv.aero.cs_ang_t.items()
                },
                cs_vel_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in dv.aero.cs_vel_t.items()
                },
                f_shape=j_shape,
            ),
        )

        n_dof: int = self.structure.n_dof
        n_aero_states: int = minimal_states_init.aero.n_states
        free_state_ix: Array = jnp.concatenate(
            [solve_dofs_arr + i * n_dof for i in range(4)]
            + [jnp.arange(5 * n_dof, 5 * n_dof + n_aero_states)]
            + [solve_dofs_arr + 4 * n_dof]
        )
        n_adj_dof = 4 * n_solve + n_aero_states + n_solve

        # compile function
        @jax.jit
        def adjoint_step(
            rev_i_ts_: int,
            d_j_d_x_: AeroelasticDesignVariables,
            adj_: Array,
            p_r_np1_p_q_n: Array,
            q_n: AeroelasticMinimalStates,
        ):
            return time_loop(
                rev_i_ts_=rev_i_ts_,
                d_j_d_x_=d_j_d_x_,
                adj_=adj_,
                p_r_np1_p_q_n=p_r_np1_p_q_n,
                q_n=q_n,
            )

        # pass through time steps backwards to obtain adjoints
        d_j_d_x, adj, p_r1_p_q0, _ = jax.lax.fori_loop(
            lower=case.structure.n_tstep - 1 - last_active_i_ts,
            upper=case.structure.n_tstep - 1,
            body_fun=lambda i_ts, args: adjoint_step(i_ts, *args),
            init_val=(
                dv_grad_init,
                jnp.zeros((case.structure.n_tstep + 1, n_j, n_adj_dof))
                if save_adjoint
                else jnp.zeros((n_j, n_adj_dof)),
                jnp.zeros((n_adj_dof, n_adj_dof)),
                case.get_minimal_states(i_ts=-1),
            ),
        )

        # solve initial timestep adjoint, as there is no r0
        p_j0_p_q0: Array
        p_j0_p_x: StructuralDesignVariables
        q0 = case.get_minimal_states(0)
        p_j0_p_q0, p_j0_p_x = jax.jacrev(
            lambda q_free, dv__: jnp.atleast_1d(
                objective(
                    self.minimal_states_to_full_states(
                        AeroelasticMinimalStates.from_vector(
                            vect=q0.ravel().at[free_state_ix].set(q_free),
                            n_dof=self.structure.n_dof,
                            aero_shapes=minimal_states_init.aero.shapes(),
                        ),
                        dv=dv__,
                    ),
                    dv__,
                    0,
                )
            ),
            argnums=(0, 1),
        )(q0.ravel()[free_state_ix], dv)

        # add initial direct sensitivity
        d_j_d_x += p_j0_p_x

        # include initial state sensitivity
        if p_varphi_p_x is not None:
            p_q0_p_x = self.compute_p_q0_p_x(
                case=case[0].to_static(),
                p_varphi_p_x=p_varphi_p_x,
                horseshoe=static_horseshoe,
                solve_dofs=solve_dofs_arr,
            )

            p_j0_p_q0_full = jnp.zeros((n_j, minimal_states_init.n_states))
            if case.structure.n_tstep > 1:
                # add on Jacobian of residual at i_ts=1 with respect to states at i_ts=0
                p_j0_p_q0_full = p_j0_p_q0_full.at[:, free_state_ix].set(
                    (adj[1, :, :] if save_adjoint else adj) @ p_r1_p_q0
                )

            # add zero terms for prescribed dofs
            p_j0_p_q0_full = p_j0_p_q0_full.at[:, free_state_ix].add(p_j0_p_q0)
            d_j_d_x += p_q0_p_x.premultiply_adj(p_j0_p_q0_full)

        # restore original shape of j, and cut off zeros for past-end timestep and initial timestep which are always 0
        adj = (
            adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[1:-1]
            if save_adjoint
            else None
        )

        return d_j_d_x, adj
