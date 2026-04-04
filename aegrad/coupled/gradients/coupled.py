from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable, TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax import Array, vmap

from structure.data_structures import OptionalJacobians

from coupled.gradients.data_structures import (
    AeroelasticStates,
    AeroelasticDesignVariables,
    AeroelasticDesignGradients,
)

from coupled.coupled import BaseCoupledAeroelastic
from aero.flowfields import Constant
from algebra.se3 import exp_se3
from structure.gradients.data_structures import StructureFullStates

if TYPE_CHECKING:
    from coupled.coupled import StaticAeroelastic

type AeroelasticObjectiveFunction = Callable[
    [AeroelasticStates, AeroelasticDesignVariables], Array
]


class CoupledAeroelastic(BaseCoupledAeroelastic):
    def _aeroelastic_states_res_from_dv_varphi(
            self,
            dv: AeroelasticDesignVariables,
            varphi: Array,
            t: Array,
            use_horseshoe: bool,
    ) -> tuple[AeroelasticStates, Array]:
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

        aero_states = aero_sol.get_full_states()

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

        return AeroelasticStates(structure=struct_states, aero=aero_states), f_res

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

        solve_dofs = jnp.setdiff1d(
            jnp.arange(self.structure.n_dof),
            case.structure.prescribed_dofs,
            size=self.structure.n_dof - case.structure.prescribed_dofs.size,
        )
        if optional_jacobians is not None:
            self.structure.optional_jacobians = optional_jacobians

        dv = self.get_design_variables(case=case)
        states = case.get_full_states()

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(states, dv))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.structure.n_x + dv.aero.n_x
        n_u_full = self.structure.n_dof

        varphi = case.structure.varphi

        # gradient of objective w.r.t. minimal states and design variables
        p_f_p_varphi, p_f_p_x = jax.jacrev(
            lambda flat_varphi, dv_: objective(
                self._aeroelastic_states_res_from_dv_varphi(
                    dv_, flat_varphi.reshape(self.structure.n_nodes, 6), t=case.aero.t,
                    use_horseshoe=case.aero.horseshoe
                )[0],
                dv_,
            ), argnums=(0, 1)
        )(varphi.ravel(), dv)  # [n_f, n_u_full], [n_f, n_x]

        # gradient of residual w.r.t. minimal states and design variables (used by linear solves)
        p_res_p_varphi, p_res_p_x = jax.jacrev(
            lambda flat_varphi, dv_: self._aeroelastic_states_res_from_dv_varphi(
                dv_, flat_varphi.reshape(self.structure.n_nodes, 6), t=case.aero.t, use_horseshoe=case.aero.horseshoe
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
