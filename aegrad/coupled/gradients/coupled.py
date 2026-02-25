from __future__ import annotations
from copy import copy, deepcopy
from typing import Optional, Callable, TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.structure import OptionalJacobians

from coupled.gradients.data_structures import (
    AeroelasticStates,
    AeroelasticDesignVariables,
    AeroelasticDesignGradients,
)
from structure import StructuralStates, StructuralDesignVariables
from aero.gradients.data_structures import AeroStates, AeroDesignVariables
from aegrad.coupled.coupled import BaseCoupledAeroelastic
from aegrad.algebra.se3 import hg_to_d
from aero.flowfields import Constant
from algebra.se3 import exp_se3

if TYPE_CHECKING:
    from aegrad.coupled.coupled import StaticAeroelastic

type AeroelasticObjectiveFunction = Callable[
    [AeroelasticStates, AeroelasticDesignVariables], Array
]


class CoupledAeroelastic(BaseCoupledAeroelastic):
    def _aeroelastic_states_res_from_dv_n(
        self,
        dv: AeroelasticDesignVariables,
        n: Array,
        t: Array,
        use_horseshoe: bool,
    ) -> tuple[AeroelasticStates, Array]:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        """

        # make a copy of the structure_dv object to prevent modifying the original states
        inner_beam = deepcopy(self.structure)

        inner_aero = deepcopy(self.uvlm)

        inner_case = CoupledAeroelastic(
            structure=inner_beam,
            aero=inner_aero,
            fsi_convergence_settings=self.fsi_convergence_settings,
            verbosity=self.verbosity,
        )

        inner_case.set_design_variables(
            coords=dv.structure.x0,
            k_cs=dv.structure.k_cs,
            m_cs=dv.structure.m_cs,
            m_lumped=dv.structure.m_lumped,
            dt=self.uvlm.dt,
            flowfield=Constant(
                u_inf=dv.aero.u_inf,
                rho=dv.aero.rho,
                relative_motion=self.uvlm.flowfield.relative_motion,
            ),  # TODO: generalise
            delta_w=self.uvlm.delta_w,
            x0_aero=dv.aero.x0_aero,
            remove_checks=True,
        )

        exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes_, 4, 4]
        hg = jnp.einsum(
            "ijk,ikl->ijl", inner_case.structure.hg0, exp_n
        )  # [n_nodes_, 4, 4]

        # evaluate aero forcing and project to beam nodes
        aero_obj = inner_case.uvlm.solve_static(hg=hg, t=t, horseshoe=use_horseshoe)
        f_ext_aero_global = aero_obj.project_forcing_to_beam(
            rmat=hg[:, :3, :3], x0_aero=self.uvlm.x0_b, include_unsteady=False
        )

        d = inner_case.structure._make_d(hg)
        p_d = inner_case.structure._make_p_d(d)
        eps = inner_case.structure._make_eps(d)
        f_int = inner_case.structure._assemble_vector_from_entries(
            inner_case.structure._make_f_int(p_d, eps)
        ).reshape(-1, 6)
        if inner_case.structure.use_gravity:
            m_t = inner_case.structure._make_m_t(d)
            f_grav = inner_case.structure._assemble_vector_from_entries(
                inner_case.structure._make_f_grav(m_t, hg[:, :3, :3])
            ).reshape(-1, 6)
        else:
            m_t = None
            f_grav = None

        if dv.structure.f_ext_dead is not None:
            f_ext_dead = inner_case.structure._make_f_dead_ext(
                dv.structure.f_ext_dead, hg[:, :3, :3]
            )
        else:
            f_ext_dead = None

        f_ext_aero = inner_case.structure._make_f_dead_ext(
            f_ext_aero_global, hg[:, :3, :3]
        )

        struct_states = StructuralStates(
            hg=hg,
            d=d,
            eps=eps,
            f_int=f_int,
            f_ext_dead=f_ext_dead,
            f_ext_aero=f_ext_aero,
            f_grav=f_grav,
        )

        aero_states = AeroStates(
            gamma_b=aero_obj.gamma_b,
            gamma_w=aero_obj.gamma_w,
            f_steady=aero_obj.f_steady,
            f_unsteady=None,
        )

        f_dead_total = inner_case.structure._make_f_ext_dead_tot(
            f_ext_dead, f_ext_aero_global, i_load_step=None, i_ts=None
        )

        f_res = inner_case.structure._make_f_res(
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
        Computes the static adjoint of the structure_dv, which is used to compute gradients of the loss with respect to
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

        # base parameters
        n = vmap(hg_to_d)(self.structure.hg0, case.structure.hg).ravel()  # [n_dof]
        d = case.structure.d
        eps = case.structure.eps

        # make copy of structure_dv which has been converted to global coordinates.
        gs = copy(case.structure)
        gs.to_global()

        # make design variables for current state of structure_dv
        struct_dv = StructuralDesignVariables(
            x0=self.structure.x0,
            k_cs=self.structure.k_cs,
            m_cs=self.structure.m_cs if self.structure.use_m_cs else None,
            m_lumped=self.structure.m_lumped
            if self.structure.use_lumped_mass
            else None,
            f_ext_follower=case.structure.f_ext_follower,
            f_ext_dead=gs.f_ext_dead,
        )

        struct_states = StructuralStates(
            hg=case.structure.hg,
            d=d,
            eps=eps,
            f_int=case.structure.f_int,
            f_ext_dead=case.structure.f_ext_dead,
            f_ext_aero=case.structure.f_ext_aero,
            f_grav=case.structure.f_grav,
        )

        aero_dv = AeroDesignVariables(
            x0_aero=self.uvlm.x0_b,
            u_inf=self.uvlm.flowfield.u_inf,
            rho=self.uvlm.flowfield.rho,
        )

        aero_states = AeroStates(
            gamma_b=case.aero.gamma_b,
            gamma_w=case.aero.gamma_w,
            f_steady=case.aero.f_steady,
            f_unsteady=None,
        )

        dv = AeroelasticDesignVariables(structure_dv=struct_dv, aero_dv=aero_dv)
        states = AeroelasticStates(structure=struct_states, aero=aero_states)

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(states, dv))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.structure.n_x + dv.aero.n_x
        n_u = len(solve_dofs)
        n_u_full = self.structure.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n = (jax.jacrev if n_f < n_u else jax.jacfwd)(
            lambda n_: objective(
                self._aeroelastic_states_res_from_dv_n(
                    dv, n_, t=case.aero.t, use_horseshoe=case.aero.horseshoe
                )[0],
                dv,
            )
        )(n).reshape(n_f, n_u_full)[:, solve_dofs]  # [n_f, n_u]

        # gradient of objective w.r.t. design variables
        p_f_p_x = (jax.jacrev if n_f < n_x else jax.jacfwd)(
            lambda dv_: objective(
                self._aeroelastic_states_res_from_dv_n(
                    dv_, n, t=case.aero.t, use_horseshoe=case.aero.horseshoe
                )[0],
                dv_,
            )
        )(dv).ravel_jacobian(n_f, n_x)  # [n_f, n_x]

        d_res_d_n = jax.jacrev(
            lambda n_: self._aeroelastic_states_res_from_dv_n(
                dv, n_, t=case.aero.t, use_horseshoe=case.aero.horseshoe
            )[1]
        )(n)[jnp.ix_(solve_dofs, solve_dofs)]  # [n_u, n_u]

        # gradient of residual w.r.t. design variables
        p_res_p_x = (jax.jacrev if n_u < n_x else jax.jacfwd)(
            lambda dv_: self._aeroelastic_states_res_from_dv_n(
                dv_, n, t=case.aero.t, use_horseshoe=case.aero.horseshoe
            )[1]
        )(dv).ravel_jacobian(n_u_full, n_x)[solve_dofs, :]  # [n_u, n_x]

        if n_f > n_x:
            # forward mode
            d_n_d_x = jnp.linalg.solve(d_res_d_n, p_res_p_x).reshape(
                self.structure.n_nodes, 6, n_x
            )  # [n_u, n_x]
            rhs = jnp.einsum("ij,ijk->k", p_f_p_n, d_n_d_x)  # [n_f, n_x]
        else:
            # reverse mode
            d_f_d_res = jnp.linalg.solve(d_res_d_n.T, p_f_p_n.T).T  # [n_f, n_u]
            rhs = d_f_d_res @ p_res_p_x  # [n_f, n_x]

        d_f_d_x_dict = dv.from_adjoint(f_shape, p_f_p_x - rhs)

        return dv.split_adjoint(d_f_d_x=d_f_d_x_dict, f_shape=f_shape)
