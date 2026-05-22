from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING, overload, Sequence, Final

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.aero.gradients.data_structures import (
    AeroDesignVariables,
    AeroStates,
    AeroGradsToCompute,
)
from aegrad.aero.utils import project_forcing_to_beam
from aegrad.algebra.array_utils import ArrayList
from aegrad.coupled import DynamicAeroelastic
from aegrad.utils.print_utils import jax_print, VerbosityLevel, verbosity
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
from aegrad.coupled.data_structures import DynamicAeroelasticSnapshot
from aegrad.coupled.gradients.data_structures import AeroelasticGradsToCompute
from aegrad.structure.gradients.data_structures import StructuralGradsToCompute
from aegrad.utils.print_utils import warn
from aegrad.coupled.gradients.data_structures import TrimVariables
from aegrad.utils.print_utils import (
    VERBOSITY_LEVEL,
    print_table_title,
)
from aegrad.utils.print_utils import print_table_line

if TYPE_CHECKING:
    from aegrad.coupled.coupled import StaticAeroelastic

type AeroelasticObjectiveFunction = (
    Callable[[AeroelasticFullStates, AeroelasticDesignVariables, None], Array]
    | Callable[[AeroelasticFullStates, AeroelasticDesignVariables, int], Array]
)
ORIENTATION_DICT: Final[dict[str, int]] = {"x": 0, "y": 1, "z": 2}


class CoupledAeroelastic(BaseCoupledAeroelastic):
    def _aeroelastic_states_res_from_dv_varphi(
        self,
        dv: AeroelasticDesignVariables,
        varphi: Array,
        thrust: dict[str, Array],
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
            m_lumped=None
            if not self.structure.use_lumped_mass
            else dv.structure.m_lumped
            if dv.structure.m_lumped is not None
            else self.structure.m_lumped,
            thrust_reference=dv.structure.thrust_t
            if dv.structure.thrust_t is not None
            else self.structure.thrust_reference,
            flowfield=self.aero.flowfield.from_design_variables(
                design_variables=dv.aero.flowfield
            )
            if dv.aero.flowfield is not None
            else self.aero.flowfield,
            delta_w=self.aero.delta_w,
            dt=self.aero.dt,
            x0_aero=dv.aero.x0_aero if dv.aero.x0_aero is not None else self.aero.x0_b,
            orientation_euler=dv.structure.orientation_euler
            if dv.structure.orientation_euler is not None
            else self.structure.orientation_euler,
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
            thrust_n=thrust,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        struct_states = StructureFullStates(
            hg=hg,
            varphi=varphi,
            eps=eps,
            f_elem=f_elem,
            f_res=f_res.reshape(-1, 6),
            v=None,
            v_dot=None,
        )

        aero_states = aero_sol.get_states(i_ts=i_ts)

        return AeroelasticFullStates(structure=struct_states, aero=aero_states), f_res

    def minimal_states_to_full_states(
        self,
        i_ts: int,
        q: AeroelasticMinimalStates,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
    ) -> AeroelasticFullStates:
        return AeroelasticFullStates(
            structure=self.structure.minimal_states_to_full_states(
                i_ts=i_ts, q=q.structure, dv=dv.structure, dv_full=dv_full.structure
            ),
            aero=q.aero,
        )

    @jax.jit(static_argnums=(0, 1, 2, 3, 4, 5))
    def static_adjoint(
        self,
        case: StaticAeroelastic,
        objective: AeroelasticObjectiveFunction,
        grads_to_compute: AeroelasticGradsToCompute = AeroelasticGradsToCompute(),
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
        :param grads_to_compute: Data structure which specifies which gradients to compute. This is used to speed up the
        adjoint solve by only computing the necessary Jacobian blocks.
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

        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)
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
                    dv=dv_,
                    varphi=flat_varphi.reshape(self.structure.n_nodes, 6),
                    thrust=case.structure.thrust,
                    t=case.aero.t,
                    i_ts=0,
                    use_horseshoe=static_horseshoe,
                )[0],
                dv_,
                None,
            ),
            argnums=(0, 1),
            allow_int=True,
        )(varphi.ravel(), dv)  # [n_f, n_u_full], [n_f, n_x]

        # gradient of residual w.r.t. minimal states and design variables (used by linear solves)
        p_res_p_varphi, p_res_p_x = jax.jacrev(
            lambda flat_varphi, dv_: self._aeroelastic_states_res_from_dv_varphi(
                dv=dv_,
                varphi=flat_varphi.reshape(self.structure.n_nodes, 6),
                thrust=case.structure.thrust,
                t=case.aero.t,
                i_ts=0,
                use_horseshoe=static_horseshoe,
            )[1],
            argnums=(0, 1),
            allow_int=True,
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
        grads_to_compute: Optional[AeroelasticGradsToCompute],
        p_varphi_p_x: Array,
        solve_dofs: Array,
        horseshoe: bool = False,
    ) -> AeroelasticDesignVariables:
        r"""
        Obtain the gradient of the initial minimal states for a dynamic aeroelastic system with respect to the design
        variables.
        :param case: StaticAeroelastic solution.
        :param grads_to_compute: AeroelasticGradsToCompute object.
        :param p_varphi_p_x: Jacobian of structural twists with respect to the design variables. This can be obtained
        from the static adjoint solver by using the forward mode, which results in this being the adjoint state.
        :param solve_dofs: Array of degree of freedom index which are solved for
        :param horseshoe: Use static_horseshoe wake.
        :return: Gradients of AeroelasticMinimalStates with respect to the design variables.
        """

        # design variables with variables that we don't require gradients omitted to speed up computations.
        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)

        # design variables with no omissions
        dv_full = self.get_design_variables(case=case, grads_to_compute=None)

        varphi = case.structure.varphi

        def minimal_states_from_varphi(
            varphi_: Array,
            dv_: AeroelasticDesignVariables,
        ) -> Array:
            inner_case = self.case_from_dv(dv=dv_)

            assert (
                dv_full.aero.cs_ang_t is not None and dv_full.aero.cs_vel_t is not None
            )

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
                    cs_ang_n={
                        k: jnp.atleast_1d(v)[0]
                        for k, v in (
                            dv_.aero.cs_ang_t
                            if dv_.aero.cs_ang_t is not None
                            else dv_full.aero.cs_ang_t
                        ).items()
                    },
                    cs_vel_n={
                        k: jnp.atleast_1d(v)[0]
                        for k, v in (
                            dv_.aero.cs_vel_t
                            if dv_.aero.cs_vel_t is not None
                            else dv_full.aero.cs_vel_t
                        ).items()
                    },
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
            minimal_states_from_varphi, argnums=(0, 1), allow_int=True
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
        dv_full: AeroelasticDesignVariables,
        thrust_t: dict[str, Array],
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
            dv_full=dv_full,
            f_aero_beam_n=q_n.structure.f_ext_aero,
            struct_obj=self.structure,
            approx_grads=approx_grads,
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
            thrust_t=thrust_t,
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
        grads_to_compute: Optional[
            AeroelasticGradsToCompute
        ] = AeroelasticGradsToCompute(),
        p_varphi_p_x: Optional[Array] = None,
        save_adjoint: bool = False,
        approx_grads: bool = True,
    ) -> tuple[AeroelasticDesignVariables, Array, Optional[Array]]:
        r"""
        Compute the adjoint of a coupled dynamic aeroelastic system.
        :param case: Dynamic aeroelastic case
        :param objective: Objective function that takes the system full states, design variables and timestep index,
        and returns an array.
        :param grads_to_compute: Specify which design variables for which to compute gradients for. If None, all
        available gradients are computed.
        :param p_varphi_p_x: Gradient of initial twists with respect to design variables. In practice, this is found
        from the static solve.
        :param save_adjoint: Whether to save the adjoint of the dynamic aeroelastic system.
        :param approx_grads: Whether to use gradient approximation or not.
        :return: Gradient of sum of objective across timesteps with respect to design variables, objective at each time
        step, and optional adjoint states.
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

        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)

        dv_full = self.get_design_variables(case=case, grads_to_compute=None)

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
        active_mask = jnp.any(j_eval, axis=-1)
        last_active_i_ts = jnp.max(jnp.where(active_mask, jnp.arange(n_tstep), 0))

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
                            i_ts=i_ts,
                            q=AeroelasticMinimalStates.from_vector(
                                vect=q_n.ravel().at[free_state_ix].set(q_free),
                                n_dof=self.structure.n_dof,
                                aero_shapes=minimal_states_init.aero.shapes(),
                            ),
                            dv=dv__,
                            dv_full=dv_full,
                        ),
                        dv__,
                        i_ts,
                    )
                ),
                argnums=(0, 1),
                allow_int=True,
            )(q_n.ravel()[free_state_ix], dv)

            # find gradients of residual function using hand-assembled block Jacobians
            p_res_p_q_nm1, p_res_p_q_n, p_v_dot_res_p_struct_dv, p_aero_res_d_dv = (
                self.timestep_residual_jacobians(
                    i_ts=i_ts,
                    t=t_n,
                    q_nm1=q_nm1,
                    q_n=q_n,
                    dv_=dv,
                    dv_full=dv_full,
                    thrust_t=case.structure.thrust,
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

        assert dv_full.aero.cs_ang_t is not None and dv_full.aero.cs_vel_t is not None
        dv_grad_init = AeroelasticDesignVariables(
            structure_dv=StructuralDesignVariables(
                x0=jnp.zeros((*j_shape, *self.structure.x0.shape))
                if grads_to_compute is None or grads_to_compute.structure.x0
                else None,
                orientation_euler=jnp.zeros((*j_shape, 3))
                if grads_to_compute is None
                or grads_to_compute.structure.orientation_euler
                else None,
                k_cs=jnp.zeros((*j_shape, *self.structure.k_cs.shape))
                if grads_to_compute is None or grads_to_compute.structure.k_cs
                else None,
                m_cs=jnp.zeros((*j_shape, *self.structure.m_cs.shape))
                if grads_to_compute is None or grads_to_compute.structure.m_cs
                else None,
                m_lumped=jnp.zeros((*j_shape, *self.structure.m_lumped.shape))
                if self.structure.use_lumped_mass
                and (grads_to_compute is None or grads_to_compute.structure.m_lumped)
                else None,
                f_ext_dead=jnp.zeros((*j_shape, *case.structure.f_ext_dead.shape))
                if case.structure.f_ext_dead is not None
                and (grads_to_compute is None or grads_to_compute.structure.f_ext_dead)
                else None,
                f_ext_follower=jnp.zeros(
                    (*j_shape, *case.structure.f_ext_follower.shape)
                )
                if case.structure.f_ext_follower is not None
                and (
                    grads_to_compute is None
                    or grads_to_compute.structure.f_ext_follower
                )
                else None,
                thrust_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in case.structure.thrust.items()
                }
                if grads_to_compute is None or grads_to_compute.structure.thrust_t
                else None,
                f_shape=j_shape,
            ),
            aero_dv=AeroDesignVariables(
                x0_aero=ArrayList(
                    [jnp.zeros((*j_shape, *arr.shape)) for arr in self.aero.x0_b]
                )
                if (grads_to_compute is None or grads_to_compute.aero.x0_aero)
                else None,
                flowfield={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in self.aero.flowfield.to_design_variables().items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.flowfield)
                else None,
                cs_ang_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in (
                        dv.aero.cs_ang_t
                        if dv.aero.cs_ang_t is not None
                        else dv_full.aero.cs_ang_t
                    ).items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.cs_ang_t)
                else None,
                cs_vel_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in (
                        dv.aero.cs_vel_t
                        if dv.aero.cs_vel_t is not None
                        else dv_full.aero.cs_vel_t
                    ).items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.cs_vel_t)
                else None,
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
        d_j_d_x: AeroelasticDesignVariables
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
                        i_ts=0,
                        q=AeroelasticMinimalStates.from_vector(
                            vect=q0.ravel().at[free_state_ix].set(q_free),
                            n_dof=self.structure.n_dof,
                            aero_shapes=minimal_states_init.aero.shapes(),
                        ),
                        dv=dv__,
                        dv_full=dv_full,
                    ),
                    dv__,
                    0,
                )
            ),
            argnums=(0, 1),
            allow_int=True,
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
                grads_to_compute=grads_to_compute,
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

        # restore gradient meta-data
        d_j_d_x.mapping = dv.mapping

        return d_j_d_x, j_eval, adj

    @overload
    def trim(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
        zero_force_dofs: Sequence[int] | Array | slice | int | None,
        trim_cs: Optional[Sequence[str]],
        thrust_nodes: Optional[Sequence[str]],
        trim_orientation: Optional[str | Sequence[str]] = ...,
        f_ext_follower: Optional[Array] = ...,
        f_ext_dead: Optional[Array] = ...,
        t: float | Array = ...,
        load_steps: int = ...,
        trim_relaxation: float = ...,
        horseshoe: bool = ...,
    ) -> tuple[StaticAeroelastic | DynamicAeroelasticSnapshot, TrimVariables]: ...

    @overload
    def trim(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
        zero_force_dofs: Sequence[int] | Array | slice | int | None,
        trim_cs: Optional[Sequence[str]],
        thrust_nodes: Optional[Sequence[str]],
        trim_orientation: Optional[str | Sequence[str]] = ...,
        f_ext_follower: Optional[Array] = ...,
        f_ext_dead: Optional[Array] = ...,
        t: float | Array = ...,
        load_steps: int = ...,
        trim_relaxation: float = ...,
        horseshoe: bool = ...,
    ) -> tuple[StaticAeroelastic | DynamicAeroelasticSnapshot, TrimVariables]: ...

    def trim(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
        zero_force_dofs: Sequence[int] | Array | slice | int | None,
        trim_cs: Optional[Sequence[str]],
        thrust_nodes: Optional[Sequence[str]],
        trim_orientation: Optional[str | Sequence[str]] = "x",
        trim_f_abs_tolerance: float = 1e-3,
        f_ext_follower: Optional[Array] = None,
        f_ext_dead: Optional[Array] = None,
        t: float | Array = 0.0,
        load_steps: int = 1,
        trim_relaxation: float = 1.0,
        horseshoe: bool = False,
    ) -> tuple[StaticAeroelastic | DynamicAeroelasticSnapshot, TrimVariables]:
        r"""
        Trim an aircraft such that the resulting sum of forces on the aircraft is zero without any supports.
        :param prescribed_dofs: Degrees of freedom which are clamped for the trim process.
        :param zero_force_dofs: Degrees freedom where we wish to drive the clamping force to zero. This is not
        necessarily the same as prescribed_dofs, as in some cases there are degrees of freedom we will allow to have a
        non-zero clamping force. For example in the case of a clamped cantilever wing where we wish to find the angle
        of attack that gives lift equal to the weight, there would be a nonzero pitching moment.
        :param trim_cs: Keys of control surfaces which are to be used to trim the aircraft.
        :param thrust_nodes: Keys of thrust nodes which are to be used to trim the aircraft.
        :param trim_orientation: Inertial axis (or axes if a sequence is provided) around which the aircraft is
        rotated about at the clamp to achieve trim.
        :param trim_f_abs_tolerance: Absolute maximum force residual at the clamped nodes for convergence to be achieved.
        :param f_ext_follower: External follower forces, [n_nodes, 6].
        :param f_ext_dead: external dead forces, [n_nodes, 6].
        :param t: Time at which to trim the aircraft, default zero.
        :param load_steps: Number of load steps used for the static solution.
        :param trim_relaxation: Relaxation factor for updates to degrees of freedom used to achieve trim.
        :param horseshoe: If true, use a horseshoe wake formulation.
        :return: Aeroelastic solution object for the trimmed aircraft.
        """

        trim_cs_: Sequence[str] = trim_cs if trim_cs is not None else []
        thrust_nodes_: Sequence[str] = thrust_nodes if thrust_nodes is not None else []
        trim_orientation_: Sequence[str] = (
            [trim_orientation]
            if isinstance(trim_orientation, str)
            else trim_orientation
            if trim_orientation is not None
            else []
        )

        zero_force_dofs_: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            zero_force_dofs
        )

        prescribed_dofs_: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )

        if not self.structure.use_gravity:
            warn("Gravity is not enabled. Trim may result in unexpected behaviour.")

        # initial set of variables
        trim_variables_init: TrimVariables = TrimVariables(
            cs_ang={k: self.aero.cs_ang0[k] for k in trim_cs_},
            thrust={k: self.structure.thrust_reference[k] for k in thrust_nodes_},
            trim_angles={
                k: self.structure.orientation_euler[ORIENTATION_DICT[k]]
                for k in trim_orientation_
            },
        )

        ae_sol_init = self.reference_configuration(
            horseshoe=horseshoe,
            prescribed_dofs=prescribed_dofs_,
            use_f_ext_dead=f_ext_dead is not None,
            use_f_ext_follower=f_ext_follower is not None,
        )

        inner_case = deepcopy(self)

        def trim_body(
            i_iter: int,
            trim_variables_: TrimVariables,
            sol_: StaticAeroelastic,
            f_clamp_: Array,
        ) -> tuple[int, TrimVariables, StaticAeroelastic, Array]:
            i_iter, _, tv, sol, fc = self.trim_iter(
                i_iter,
                inner_case,
                trim_variables_,
                sol_,
                f_clamp_,
                prescribed_dofs=prescribed_dofs_,
                zero_force_dofs=zero_force_dofs_,
                f_ext_dead=f_ext_dead,
                f_ext_follower=f_ext_follower,
                t=jnp.array(t),
                load_steps=load_steps,
                horseshoe=horseshoe,
                trim_cs=trim_cs_,
                thrust_nodes=thrust_nodes_,
                trim_orientation=trim_orientation_,
                trim_relaxation=trim_relaxation,
            )
            return i_iter, tv, sol, fc

        # run trim loop
        print_table_title(title="Trim", inner_width=81)
        n_iter, trim_variables, ae_sol, f_clamp = jax.lax.while_loop(
            lambda args_: jnp.any(jnp.abs(args_[3]) >= trim_f_abs_tolerance),
            body_fun=lambda args_: trim_body(*args_),
            init_val=(
                0,
                trim_variables_init,
                ae_sol_init,
                jnp.full((len(zero_force_dofs_)), 1e10),
            ),
        )
        print_table_line(inner_width=81)

        new_orientation: Array = self.structure.orientation_euler
        for k, v in trim_variables.trim_angles.items():
            new_orientation = new_orientation.at[ORIENTATION_DICT[k]].set(v)

        # set solutions into case object
        self.set_design_variables(
            coords=self.structure.x0,
            k_cs=self.structure.k_cs,
            m_cs=self.structure.m_cs,
            m_lumped=self.structure.m_lumped
            if self.structure.use_lumped_mass
            else None,
            dt=self.aero.dt,
            flowfield=self.aero.flowfield,
            delta_w=self.aero.delta_w,
            x0_aero=self.aero.x0_b,
            thrust_reference=self.structure.thrust_reference | trim_variables.thrust,
            orientation_euler=new_orientation,
            cs_angles_reference=self.aero.cs_ang0 | trim_variables.cs_ang,
            remove_checks=True,
        )

        return ae_sol, trim_variables

    def trim_angles_to_euler(self, trim_angles: dict[str, Array]) -> Array:
        r"""
        Find the 3 Euler angles describing the aircraft orientation. This allows for any combination to be set by
        the trim routine, with values not passed using the fixed values provided in the reference orientation.
        :param trim_angles: Dictionary of axis-angle pairs.
        :return: Euler angles describing the aircraft orientation, [3]
        """

        orientation_euler = self.structure.orientation_euler
        for k, v in trim_angles.items():
            orientation_euler = orientation_euler.at[ORIENTATION_DICT[k]].set(v)
        return orientation_euler

    @staticmethod
    def trim_iter(
        i_iter: int,
        inner_case: CoupledAeroelastic,
        trim_variables: TrimVariables,
        _: StaticAeroelastic,
        __: Array,
        *,
        prescribed_dofs: tuple[int, ...],
        zero_force_dofs: tuple[int, ...],
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        t: Array,
        load_steps: int,
        trim_relaxation: float,
        horseshoe: bool,
        trim_cs: Sequence[str],
        thrust_nodes: Sequence[str],
        trim_orientation: Sequence[str],
    ) -> tuple[int, CoupledAeroelastic, TrimVariables, StaticAeroelastic, Array]:

        inner_case.set_design_variables(
            coords=inner_case.structure.x0,
            k_cs=inner_case.structure.k_cs,
            m_cs=inner_case.structure.m_cs,
            m_lumped=inner_case.structure.m_lumped
            if inner_case.structure.use_lumped_mass
            else None,
            thrust_reference=inner_case.structure.thrust_reference
            | trim_variables.thrust,
            dt=inner_case.aero.dt,
            flowfield=inner_case.aero.flowfield,
            delta_w=inner_case.aero.delta_w,
            x0_aero=inner_case.aero.x0_b,
            orientation_euler=inner_case.trim_angles_to_euler(
                trim_variables.trim_angles
            ),
            cs_angles_reference=inner_case.aero.cs_ang0 | trim_variables.cs_ang,
            remove_checks=True,
        )

        # avoid printing structure messages if Verbosity is NORMAL or lower
        with verbosity(
            level=VerbosityLevel.SILENT
            if VERBOSITY_LEVEL.value <= VerbosityLevel.NORMAL.value
            else VERBOSITY_LEVEL
        ):
            ae_sol = inner_case.static_solve(
                prescribed_dofs=prescribed_dofs,
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                t=t,
                load_steps=load_steps,
                horseshoe=horseshoe,
            )

        f_clamp = ae_sol.structure.f_res.ravel()[jnp.array(zero_force_dofs)]

        def objective(states: AeroelasticFullStates, *_) -> Array:
            return states.structure.f_res.ravel()[jnp.array(zero_force_dofs)]

        d_f_clamp_d_x: AeroelasticDesignVariables
        adj: Array
        d_f_clamp_d_x, adj = inner_case.static_adjoint(
            case=ae_sol,
            objective=objective,
            grads_to_compute=AeroelasticGradsToCompute(
                structure=StructuralGradsToCompute(
                    x0=False,
                    orientation_euler=True,
                    k_cs=False,
                    m_cs=False,
                    m_lumped=False,
                    f_ext_follower=False,
                    f_ext_dead=False,
                    thrust_t=True,
                ),
                aero=AeroGradsToCompute(
                    x0_aero=False, flowfield=False, cs_ang_t=True, cs_vel_t=False
                ),
            ),
        )

        assert (
            d_f_clamp_d_x.aero.cs_ang_t is not None
            and d_f_clamp_d_x.structure.thrust_t is not None
            and d_f_clamp_d_x.structure.orientation_euler is not None
        )
        d_f_clamp_d_cs_ang = (
            jnp.concatenate([d_f_clamp_d_x.aero.cs_ang_t[k] for k in trim_cs], axis=1)
            if trim_cs
            else jnp.zeros((len(zero_force_dofs), 0))
        )
        d_f_clamp_d_thrust = (
            jnp.concatenate(
                [d_f_clamp_d_x.structure.thrust_t[k] for k in thrust_nodes], axis=1
            )
            if thrust_nodes
            else jnp.zeros((len(zero_force_dofs), 0))
        )
        d_f_clamp_d_trim_angles = (
            jnp.concatenate(
                [
                    d_f_clamp_d_x.structure.orientation_euler[:, [ORIENTATION_DICT[k]]]
                    for k in trim_orientation
                ],
                axis=1,
            )
            if trim_orientation
            else jnp.zeros((len(zero_force_dofs), 0))
        )

        d_f_clamp_d_trim_variables = jnp.concatenate(
            [d_f_clamp_d_cs_ang, d_f_clamp_d_thrust, d_f_clamp_d_trim_angles],
            axis=1,
        )  # note that this is not square [n_zero_force_dof, n_free_trim_dof]

        trim_update = (
            jnp.linalg.lstsq(a=d_f_clamp_d_trim_variables, b=f_clamp)[0]
            * trim_relaxation
        )  # subtract this from current solution to update

        # update trim_variables
        idx = 0
        updated_cs_ang = {}
        for k in trim_cs:
            updated_cs_ang[k] = trim_variables.cs_ang[k] - trim_update[idx]
            idx += 1
        updated_thrust = {}
        for k in thrust_nodes:
            updated_thrust[k] = trim_variables.thrust[k] - trim_update[idx]
            idx += 1
        updated_trim_angles = {}
        for k in trim_orientation:
            updated_trim_angles[k] = trim_variables.trim_angles[k] - trim_update[idx]
            idx += 1

        trim_variables = TrimVariables(
            cs_ang=updated_cs_ang,
            thrust=updated_thrust,
            trim_angles=updated_trim_angles,
        )

        trim_variables.print(i_iter=i_iter, f_clamp=f_clamp)

        return i_iter + 1, inner_case, trim_variables, ae_sol, f_clamp
