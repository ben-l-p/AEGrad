from __future__ import annotations
from copy import deepcopy
from typing import (
    Any,
    Optional,
    Callable,
    TYPE_CHECKING,
    overload,
    Sequence,
    Final,
)

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
from aegrad.coupled.gradients.data_structures import (
    AeroelasticGradsToCompute,
    AeroelasticJacobianApproximations,
)
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
            x0_aero=dv.aero.x0_b if dv.aero.x0_b is not None else self.aero.x0_b,
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

        jax_print("Computing static adjoint", verbose_level=VerbosityLevel.NORMAL)

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

    @jax.jit(static_argnums=(0, 1, 2, 3, 7))
    def compute_p_j0_p_x(
        self,
        case: StaticAeroelastic,
        objective: AeroelasticObjectiveFunction,
        grads_to_compute: Optional[AeroelasticGradsToCompute],
        p_varphi_p_x: Optional[Array],
        solve_dofs: Array,
        future_cot_q0_full: Array,
        horseshoe: bool = False,
    ) -> AeroelasticDesignVariables:
        r"""
        Compute the initial-timestep contribution to ``d_j_d_x``, namely ``d j_0/d x + future_cot_q0 @ d q_0/d x``,
        with inclusion of initial deformation sensitivity included if ``p_varphi_p_x`` is passed.
        :param case: StaticAeroelastic solution for which to obtain the gradient.
        :param objective: Objective function which takes the system full states, design variables and timestep index.
        :param grads_to_compute: Grads to compute when computing design gradient.
        :param p_varphi_p_x: Optional Jacobian to account for sensitivity of initial deformation to design variables.
        :param solve_dofs: Array of degree of freedom indices which are solved for.
        :param future_cot_q0_full: Cotangent on the initial minimal states from the time-loop adjoint
        (i.e. ``adj_1 @ p_r_1/p_q_0`` for free DOFs, zero on prescribed DOFs), shape ``(n_j, n_q0)``.
        :param horseshoe: Flag for using horseshoe wake.
        :return: Initial-timestep contribution to the total design-variable gradient.
        """

        # design variables with variables that we don't require gradients omitted to speed up computations.
        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)

        # design variables with no omissions
        dv_full = self.get_design_variables(case=case, grads_to_compute=None)

        varphi = case.structure.varphi

        def objective_from_varphi(
            varphi_: Array,
            dv_: AeroelasticDesignVariables,
        ) -> tuple[Array, Array]:
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
                    hg_nm1=None,
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
                    cs_ang_nm1=None,
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

            q0 = AeroelasticMinimalStates(structure=q_structure, aero=q_aero).ravel()
            q0_full = self.minimal_states_to_full_states(
                i_ts=0,
                q=AeroelasticMinimalStates.from_vector(
                    vect=q0,
                    n_dof=n_dof,
                    aero_shapes=aero_shapes,
                ),
                dv=dv_,
                dv_full=dv_full,
            )
            j0 = jnp.atleast_1d(objective(q0_full, dv_, 0))
            return j0, q0

        n_j = future_cot_q0_full.shape[0]
        f_shape = (n_j,)
        n_dof = self.structure.n_dof
        aero_shapes = case.aero.get_states(i_ts=0).shapes()

        _, vjp_fn = jax.vjp(objective_from_varphi, varphi, dv)
        adj_varphi, adj_dv_raw = jax.vmap(vjp_fn)((jnp.eye(n_j), future_cot_q0_full))

        # direct term
        adj_dv = AeroelasticDesignVariables(
            structure_dv=StructuralDesignVariables(
                **{k: getattr(adj_dv_raw.structure, k) for k in dv.structure.to_dict()},
                f_shape=f_shape,
            ),
            aero_dv=AeroDesignVariables(
                **{k: getattr(adj_dv_raw.aero, k) for k in dv.aero.to_dict()},
                f_shape=f_shape,
            ),
        )

        # indirect term
        if p_varphi_p_x is not None:
            indirect_mat = (
                adj_varphi.reshape(n_j, -1)[:, solve_dofs] @ p_varphi_p_x
            )  # (n_j, n_x)
            indirect_dict = dv.from_adjoint(f_shape, indirect_mat)
            indirect_term = AeroelasticDesignVariables(
                structure_dv=StructuralDesignVariables(
                    **{k: indirect_dict[k] for k in dv.structure.to_dict()},
                    f_shape=f_shape,
                ),
                aero_dv=AeroDesignVariables(
                    **{k: indirect_dict[k] for k in dv.aero.to_dict()},
                    f_shape=f_shape,
                ),
            )
            adj_dv += indirect_term

        return adj_dv

    def timestep_residual_jacobians(
        self,
        i_ts: int | Array,
        t: Array,
        q_nm1: AeroelasticMinimalStates,
        q_n: AeroelasticMinimalStates,
        dv_: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
        n_profile_loops: Optional[int],
        jac_options: dict,
    ) -> tuple[
        Array,
        Array,
        StructuralDesignVariables,
        AeroelasticDesignVariables,
        Optional[dict[str, dict[str, float]]],
        Optional[dict[str, dict[str, float]]],
    ]:

        if q_n.structure.f_ext_aero is None or q_nm1.structure.f_ext_aero is None:
            raise ValueError("Missing aerodynamic forcing states")

        (
            p_aero_res_p_q_aero_nm1,
            p_aero_res_p_q_aero_n,
            p_aero_res_d_dv,
            p_aero_res_p_q_struct_nm1,
            p_aero_res_p_q_struct_n,
            aero_compile_time,
            aero_run_time,
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
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
        )

        n_aero_dof, n_struct_dof = p_aero_res_p_q_struct_nm1.shape

        (
            p_struct_res_p_q_struct_nm1,
            p_struct_res_p_q_struct_n,
            p_v_dot_res_p_struct_dv,
            p_v_dot_res_p_f_ext_nm1,
            p_v_dot_res_p_f_ext_n,
            struct_compile_time,
            struct_run_time,
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
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
        )

        assert p_v_dot_res_p_f_ext_nm1 is not None and p_v_dot_res_p_f_ext_n is not None

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

        if n_profile_loops is not None:
            assert (
                aero_compile_time is not None
                and aero_run_time is not None
                and struct_compile_time is not None
                and struct_run_time is not None
            )
            compile_time = aero_compile_time | struct_compile_time
            run_time = aero_run_time | struct_run_time
        else:
            compile_time = None
            run_time = None
        return (
            p_res_p_q_nm1,
            p_res_p_q_n,
            p_v_dot_res_p_struct_dv,
            p_aero_res_d_dv,
            compile_time,
            run_time,
        )

    def construct_approximate_jacobians(
        self,
        sol: DynamicAeroelastic,
        jacobian_approximations: AeroelasticJacobianApproximations,
    ) -> dict[str, dict[str, Optional[Callable[..., Any]]]]:
        r"""
        Compute approximations for Jacobians which are specified in the jacobian_approximations data structure. The
        aerodynamic residual approximations are delegated to :meth:`UVLM.construct_approximate_jacobians`, and the
        structural residual approximations to :meth:`BeamStructure.construct_approximate_jacobians`. The two
        dictionaries are merged into a single result.
        :param sol: Solution for which approximations will be created for the initial time step.
        :param jacobian_approximations: Data structure which defines which approximations to create.
        :return: Dictionary of approximations keyed by residual name.
        """
        dv = self.get_design_variables(case=sol, grads_to_compute=None)
        solve_dofs = tuple(
            int(i)
            for i in get_solve_dofs(
                n_dof=self.structure.n_dof,
                prescribed_dofs=sol.structure.prescribed_dofs,
            )
        )

        aero_options = self.aero.construct_approximate_jacobians(
            aero_sol=sol.aero,
            structure_sol=sol.structure,
            struct_obj=self.structure,
            dv=dv,
            dv_full=dv,
            solve_dofs=solve_dofs,
            jacobian_approximations=jacobian_approximations.aero,
        )

        struct_options = self.structure.construct_approximate_jacobians(
            sol=sol.structure,
            jacobian_approximations=jacobian_approximations.structure,
        )

        return aero_options | struct_options

    def dynamic_adjoint(
        self,
        case: DynamicAeroelastic,
        objective: AeroelasticObjectiveFunction,
        jacobian_approximations: AeroelasticJacobianApproximations = AeroelasticJacobianApproximations(),
        grads_to_compute: Optional[
            AeroelasticGradsToCompute
        ] = AeroelasticGradsToCompute(),
        p_varphi_p_x: Optional[Array] = None,
        save_adjoint: bool = False,
        approx_grads: bool = True,
        n_parallel_steps: int = 1,
    ) -> tuple[AeroelasticDesignVariables, Array, Optional[Array]]:
        r"""
        Compute the adjoint of a coupled dynamic aeroelastic system.
        :param case: Dynamic aeroelastic case
        :param objective: Objective function that takes the system full states, design variables and timestep index,
        and returns an array.
        :param jacobian_approximations: Data structure which specifies Jacobian approximations to use for each part of
        the problem.
        :param grads_to_compute: Specify which design variables for which to compute gradients for. If None, all
        available gradients are computed.
        :param p_varphi_p_x: Gradient of initial twists with respect to design variables. In practice, this is found
        from the static solve.
        :param save_adjoint: Whether to save the adjoint of the dynamic aeroelastic system.
        :param approx_grads: Whether to use gradient approximation or not.
        :param n_parallel_steps: Number of time steps whose residual and objective Jacobians are evaluated together in
        a single vmap call before the adjoint linear solves are stepped sequentially across them. Larger values
        expose more parallelism but linearly increase peak memory.
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
            lambda: jnp.atleast_1d(objective(full_states_init, dv, 0))
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
        jax_print(
            "Evaluating adjoint problem from timestep 0 to {last_active_i_ts}",
            last_active_i_ts=last_active_i_ts,
            verbose_level=VerbosityLevel.NORMAL,
        )

        jac_options = self.construct_approximate_jacobians(
            sol=case, jacobian_approximations=jacobian_approximations
        )

        @jax.jit
        def objective_jacobians(
            i_ts: int, q_n: AeroelasticMinimalStates
        ) -> tuple[Array, AeroelasticDesignVariables]:
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
            return p_j_n_p_q_n, p_j_n_p_x

        def time_loop(
            rev_i_ts_: int,
            d_j_d_x_: AeroelasticDesignVariables,
            adj_np1: Array,
            p_aero_res_d_dv: AeroelasticDesignVariables,
            p_v_dot_res_p_struct_dv: StructuralDesignVariables,
            p_r_n_p_q_n: Array,
            p_r_np1_p_q_n: Array,
            p_j_n_p_x: AeroelasticDesignVariables,
            p_j_n_p_q_n: Array,
        ) -> tuple[AeroelasticDesignVariables, Array]:
            r"""
            Function to update the adjoint solution for a single timestep.
            :param rev_i_ts_: Reversed timestep index. JAX loop does not allow for reverse indexing, and so this is.
            explicitly reversed within the function body to obtain i_ts.
            :param d_j_d_x_: Design gradient to accumulate.
            :param adj_np1: Adjoint vector for n+1 time step.
            :param p_aero_res_d_dv: Jacobian of aerodynamic residual with respect to aeroelastic design variables.
            :param p_v_dot_res_p_struct_dv: Jacobian of `v_dot` residual with respect to structural design variables.
            :param p_r_n_p_q_n: Jacobian of current residual with respect to current degrees of freedom.
            :param p_r_np1_p_q_n: Jacobian of future residual with respect to current degrees of freedom.
            :param p_j_n_p_x: Jacobian of current objective with respect to aeroelastic design variables.
            :param p_j_n_p_q_n: Jacobian of current objective with respect to current degrees of freedom.
            :return: Updated design gradient `d_j_d_x_` and adjoint vector `adj_n` for current time step.
            """

            i_ts = (
                n_tstep - rev_i_ts_ - 1
            )  # index for timestep varphi, which decrements

            # solve for adjoint at current timestep
            b: Array = -(p_j_n_p_q_n.reshape(n_j, -1) + adj_np1 @ p_r_np1_p_q_n).T

            adj_n = jnp.linalg.solve(
                p_r_n_p_q_n.T,
                b,
            ).T

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

            return d_j_d_x_, adj_n

        def multi_timestep_loop(
            rev_i_ts_start: int,
            d_j_d_x_: AeroelasticDesignVariables,
            adj_np1: Array,
            p_r_np1_p_q_n: Array,
            n_parallel_steps_: int,
        ) -> tuple[AeroelasticDesignVariables, Array, Array, Optional[Array]]:
            r"""
            Routine to step the adjoint solution by multiple time steps. The residual and objective Jacobians for the
            ``n_parallel_steps`` time steps in the chunk are evaluated in a single :func:`jax.vmap` call so that the
            (often dominant) Jacobian assemblies execute in parallel. The adjoint linear solves are then performed
            sequentially backwards in time over the precomputed Jacobians, because they are coupled through the
            adjoint carry. Chunking keeps the peak memory bounded by ``n_parallel_steps`` worth of Jacobians rather
            than the full ``n_tstep`` history.
            :param rev_i_ts_start: Reversed timestep index of the first (latest in real time) step in the chunk.
            ``rev_i_ts = 0`` corresponds to ``i_ts = n_tstep - 1``.
            :param d_j_d_x_: Design gradient to accumulate.
            :param adj_np1: Adjoint vector for one past the final parallel step.
            :param p_r_np1_p_q_n: Jacobian of one past the final parallel residual with respect to the final parallel
            degrees of freedom.
            :param n_parallel_steps_: Number of time steps to vmap across for obtaining Jacobians. This is kept as an
            argument to allow for the case where the total number of timesteps does not nicely divide into the chunks
            and results in a final, smaller chunk.
            :return: Updated design gradients, the adjoint vector for the first parallel time step (i.e. the earliest
            in real time within the chunk), the residual Jacobian ``p_res/p_q_{n-1}`` evaluated at that earliest
            step — to be threaded as ``p_r_np1_p_q_n`` into the next chunk — and optionally the full adjoint solution
            time history for the chunk (shape ``(n_parallel_steps, n_j, n_adj_dof)``, with index 0 at the latest real
            time).
            """

            # i_ts for each step in the chunk; index 0 is the latest in real time (processed first by the adjoint)
            i_ts_chunk = n_tstep - 1 - rev_i_ts_start - jnp.arange(n_parallel_steps_)

            def per_step_jacobians(
                i_ts: Array,
            ) -> tuple[
                Array,
                Array,
                StructuralDesignVariables,
                AeroelasticDesignVariables,
                Array,
                AeroelasticDesignVariables,
            ]:
                i_ts_nm1 = jnp.maximum(i_ts - 1, 0)
                q_nm1 = case.get_minimal_states(i_ts=i_ts_nm1)
                q_n = case.get_minimal_states(i_ts=i_ts)
                (
                    p_res_p_q_nm1,
                    p_res_p_q_n,
                    p_v_dot_res_p_struct_dv_,
                    p_aero_res_d_dv_,
                    *_,
                ) = self.timestep_residual_jacobians(
                    i_ts=i_ts,
                    t=case.structure.t[i_ts],
                    q_nm1=q_nm1,
                    q_n=q_n,
                    dv_=dv,
                    dv_full=dv_full,
                    thrust_t=case.structure.thrust,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                    n_profile_loops=None,
                    jac_options=jac_options,
                )
                p_j_n_p_q_n_, p_j_n_p_x_ = objective_jacobians(i_ts=i_ts, q_n=q_n)
                return (
                    p_res_p_q_nm1,
                    p_res_p_q_n,
                    p_v_dot_res_p_struct_dv_,
                    p_aero_res_d_dv_,
                    p_j_n_p_q_n_,
                    p_j_n_p_x_,
                )

            (
                p_res_p_q_nm1_batch,
                p_res_p_q_n_batch,
                p_v_dot_res_p_struct_dv_batch,
                p_aero_res_d_dv_batch,
                p_j_n_p_q_n_batch,
                p_j_n_p_x_batch,
            ) = jax.vmap(per_step_jacobians)(i_ts_chunk)

            n_adj = adj_np1.shape[-1]
            adj_history_init = (
                jnp.zeros((n_parallel_steps_, n_j, n_adj))
                if save_adjoint
                else jnp.zeros(())
            )

            def chunk_body(
                k: int,
                carry: tuple[AeroelasticDesignVariables, Array, Array, Array],
            ) -> tuple[AeroelasticDesignVariables, Array, Array, Array]:
                d_j_d_x_carry, adj_carry, p_r_np1_p_q_n_carry, adj_hist_carry = carry

                d_j_d_x_carry, adj_n = time_loop(
                    rev_i_ts_=rev_i_ts_start + k,
                    d_j_d_x_=d_j_d_x_carry,
                    adj_np1=adj_carry,
                    p_aero_res_d_dv=jax.tree.map(lambda x: x[k], p_aero_res_d_dv_batch),
                    p_v_dot_res_p_struct_dv=jax.tree.map(
                        lambda x: x[k], p_v_dot_res_p_struct_dv_batch
                    ),
                    p_r_n_p_q_n=p_res_p_q_n_batch[k],
                    p_r_np1_p_q_n=p_r_np1_p_q_n_carry,
                    p_j_n_p_x=jax.tree.map(lambda x: x[k], p_j_n_p_x_batch),
                    p_j_n_p_q_n=p_j_n_p_q_n_batch[k],
                )

                if save_adjoint:
                    adj_hist_carry = adj_hist_carry.at[k].set(adj_n)

                # the next iteration steps further backwards: this step's p_res/p_q_{n-1} becomes the next step's
                # p_r_np1_p_q_n
                return (
                    d_j_d_x_carry,
                    adj_n,
                    p_res_p_q_nm1_batch[k],
                    adj_hist_carry,
                )

            d_j_d_x_, adj_first, p_r_np1_p_q_n_first, adj_history = jax.lax.fori_loop(
                lower=0,
                upper=n_parallel_steps_,
                body_fun=chunk_body,
                init_val=(d_j_d_x_, adj_np1, p_r_np1_p_q_n, adj_history_init),
            )

            return (
                d_j_d_x_,
                adj_first,
                p_r_np1_p_q_n_first,
                adj_history if save_adjoint else None,
            )

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
                x0_b=ArrayList(
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

        # pass through time steps backwards in chunks of n_parallel_steps
        n_active_steps: int = case.structure.n_tstep - 1
        n_full_chunks: int = n_active_steps // n_parallel_steps
        remainder: int = n_active_steps - n_full_chunks * n_parallel_steps

        adj_full_init: Optional[Array] = (
            jnp.zeros((case.structure.n_tstep + 1, n_j, n_adj_dof))
            if save_adjoint
            else None
        )

        def chunk_step(
            chunk_i: int,
            carry: tuple[AeroelasticDesignVariables, Array, Array, Array],
        ) -> tuple[AeroelasticDesignVariables, Array, Array, Array]:
            d_j_d_x_, adj_np1, p_r_np1_p_q_n, adj_full_ = carry
            rev_i_ts_start = chunk_i * n_parallel_steps
            d_j_d_x_, adj_np1, p_r_np1_p_q_n, adj_history = multi_timestep_loop(
                rev_i_ts_start=rev_i_ts_start,
                d_j_d_x_=d_j_d_x_,
                adj_np1=adj_np1,
                p_r_np1_p_q_n=p_r_np1_p_q_n,
                n_parallel_steps_=n_parallel_steps,
            )
            if save_adjoint:
                # adj_history is ordered with index 0 at the latest real time in the chunk
                i_ts_chunk = n_tstep - 1 - rev_i_ts_start - jnp.arange(n_parallel_steps)
                adj_full_ = adj_full_.at[i_ts_chunk].set(adj_history)
            return d_j_d_x_, adj_np1, p_r_np1_p_q_n, adj_full_

        d_j_d_x: AeroelasticDesignVariables
        d_j_d_x, adj_last, p_r1_p_q0, adj_full = jax.lax.fori_loop(
            lower=0,
            upper=n_full_chunks,
            body_fun=chunk_step,
            init_val=(
                dv_grad_init,
                jnp.zeros((n_j, n_adj_dof)),
                jnp.zeros((n_adj_dof, n_adj_dof)),
                adj_full_init,
            ),
        )

        # trailing chunk for the remaining steps
        if remainder > 0:
            rev_i_ts_start_rem = n_full_chunks * n_parallel_steps
            d_j_d_x, adj_last, p_r1_p_q0, adj_history_rem = multi_timestep_loop(
                rev_i_ts_start=rev_i_ts_start_rem,
                d_j_d_x_=d_j_d_x,
                adj_np1=adj_last,
                p_r_np1_p_q_n=p_r1_p_q0,
                n_parallel_steps_=remainder,
            )
            if save_adjoint:
                i_ts_chunk_rem = (
                    n_tstep - 1 - rev_i_ts_start_rem - jnp.arange(remainder)
                )
                adj_full = adj_full.at[i_ts_chunk_rem].set(adj_history_rem)

        adj = adj_full if save_adjoint else adj_last

        # solve initial timestep adjoint, as there is no r0
        future_cot_q0_full = jnp.zeros((n_j, minimal_states_init.n_states))
        if case.structure.n_tstep > 1:
            future_cot_q0_full = future_cot_q0_full.at[:, free_state_ix].set(
                (adj[1, :, :] if save_adjoint else adj) @ p_r1_p_q0
            )

        d_j_d_x += self.compute_p_j0_p_x(
            case=case[0].to_static(),
            objective=objective,
            grads_to_compute=grads_to_compute,
            p_varphi_p_x=p_varphi_p_x,
            solve_dofs=solve_dofs_arr,
            future_cot_q0_full=future_cot_q0_full,
            horseshoe=static_horseshoe,
        )

        # restore original shape of j, and cut off zeros for past-end timestep and initial timestep which are always 0
        adj = (
            adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[1:-1]
            if save_adjoint
            else None
        )

        # restore gradient meta-data
        d_j_d_x.mapping = dv.mapping

        return d_j_d_x, j_eval, adj

    def dynamic_adjoint_profile(
        self,
        case: DynamicAeroelastic,
        approx_grads: bool,
        jacobian_approximations: AeroelasticJacobianApproximations = AeroelasticJacobianApproximations(),
        grads_to_compute: Optional[AeroelasticGradsToCompute] = None,
        i_ts: int = 1,
        n_profile_loops: int = 10,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        r"""
        Function to time evaluation of the Jacobians used for the coupled aeroelastic adjoint solution. The structural
        residual Jacobians are profiled by delegating to :meth:`BeamStructure.dynamic_adjoint_profile`, and the
        aerodynamic residual Jacobians (bound circulation, wake propagation, bound circulation rate, aerodynamic
        forcing) are profiled here.
        :param case: Dynamic aeroelastic case from which to extract states.
        :param approx_grads: If True, neglect small gradient terms.
        :param jacobian_approximations: Define which blocks of the adjoint Jacobians will be substituted for
        approximations.
        :param grads_to_compute: AeroelasticGradsToCompute object describing which design gradients to compute. If
        None, all gradients will be computed.
        :param i_ts: Time step index where to evaluate residual Jacobians.
        :param n_profile_loops: Number of times to loop the Jacobian evaluation time for averaging the runtime.
        :return: Dictionary of {residual_name: {gradient_argument: val}} for compile time and run time respectively.
        """

        print_table_title(inner_width=95, title="Aeroelastic Adjoint Profile")

        # compute Jacobian approximations, if requested
        jac_options = self.construct_approximate_jacobians(
            sol=case, jacobian_approximations=jacobian_approximations
        )

        *_, compile_time, run_time = self.timestep_residual_jacobians(
            i_ts=i_ts,
            t=case.aero.t[i_ts],
            q_nm1=case.get_minimal_states(i_ts=i_ts - 1),
            q_n=case.get_minimal_states(i_ts=i_ts),
            dv_=self.get_design_variables(case=case, grads_to_compute=grads_to_compute),
            dv_full=self.get_design_variables(case=case, grads_to_compute=None),
            thrust_t=case.structure.thrust,
            solve_dofs=get_solve_dofs(
                n_dof=self.structure.n_dof,
                prescribed_dofs=case.structure.prescribed_dofs,
            ),
            approx_grads=approx_grads,
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
        )

        print_table_line(inner_width=95)

        return compile_time, run_time

    @overload
    def trim(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
        zero_force_dofs: Sequence[int] | Array | slice | int | None,
        trim_cs: Optional[Sequence[str]],
        thrust_nodes: Optional[Sequence[str]],
        trim_orientation: Optional[str | Sequence[str]] = ...,
        trim_f_abs_tolerance: float = 1e-3,
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
        trim_f_abs_tolerance: float = 1e-3,
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
