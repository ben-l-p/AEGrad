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

        inner_aero = deepcopy(self.aero)

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

        exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes_, 4, 4]
        hg = jnp.einsum(
            "ijk,ikl->ijl", inner_case.structure.hg0, exp_n
        )  # [n_nodes_, 4, 4]

        # evaluate aero forcing and project to beam nodes
        aero_obj = inner_case.aero.solve_static(hg=hg, t=t, horseshoe=use_horseshoe)
        f_ext_aero_global = aero_obj.project_forcing_to_beam(
            i_ts=0, rmat=hg[:, :3, :3], x0_aero=self.aero.x0_b, include_unsteady=False
        )

        d = inner_case.structure.make_d(hg)
        p_d = inner_case.structure.make_p_d(d)
        eps = inner_case.structure.make_eps(d)
        f_int = inner_case.structure.assemble_vector_from_entries(
            inner_case.structure.make_f_int(p_d, eps)
        ).reshape(-1, 6)
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

        struct_states = StructuralStates(
            hg=hg,
            eps=eps,
            f_int=f_int,
        )

        aero_states = AeroStates(
            f_steady=aero_obj.f_steady,
            f_unsteady=None,
        )

        f_dead_total = inner_case.structure.make_f_ext_dead_tot(
            f_ext_dead, f_ext_aero_global, i_load_step=None, i_ts=None
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
            eps=eps,
            f_int=case.structure.f_int,
        )

        aero_dv = AeroDesignVariables(
            x0_aero=self.aero.x0_b,
            u_inf=self.aero.flowfield.u_inf,
            rho=self.aero.flowfield.rho,
        )

        aero_states = AeroStates(
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

        # gradient of objective w.r.t. minimal states. Not overly expensive
        p_f_p_n = jax.jacrev(
            lambda n_: objective(
                self._aeroelastic_states_res_from_dv_n(
                    dv, n_, t=case.aero.t, use_horseshoe=case.aero.horseshoe
                )[0],
                dv,
            )
        )  # [n_u] -> [n_f, n_u]

        # gradient of objective w.r.t. design variables. Not overly expensive
        p_f_p_x = jax.jacrev(
            lambda dv_: objective(
                self._aeroelastic_states_res_from_dv_n(
                    dv_, n, t=case.aero.t, use_horseshoe=case.aero.horseshoe
                )[0],
                dv_,
            )
        )  # [n_x] -> [n_f, n_x]

        # gradient of residual w.r.t. design variables (used by linear solves)
        p_res_p_x = (jax.jacrev if n_u < n_x else jax.jacfwd)(
            lambda dv_: self._aeroelastic_states_res_from_dv_n(
                dv_, n, t=case.aero.t, use_horseshoe=case.aero.horseshoe
            )[1]
        )  # [n_x] -> [n_u, n_x]

        # gradient of residual w.r.t. design variables
        d_res_d_n = jax.jacfwd(
            lambda n_: self._aeroelastic_states_res_from_dv_n(
                dv, n_, t=case.aero.t, use_horseshoe=case.aero.horseshoe
            )[1]
        )  # [n_u] -> [n_u, n_u]

        adj = (
            jnp.linalg.solve(
                d_res_d_n(n)[jnp.ix_(solve_dofs, solve_dofs)].T,
                p_f_p_n(n).reshape(n_f, -1)[:, solve_dofs].T,
            ).T
            @ p_res_p_x(dv).ravel_jacobian(f_size=n_u_full, x_size=n_x)[solve_dofs, :]
        )

        d_f_d_x_dict = dv.from_adjoint(
            f_shape, p_f_p_x(dv).ravel_jacobian(f_size=n_f, x_size=n_x) - adj
        )

        return dv.split_adjoint(d_f_d_x=d_f_d_x_dict, f_shape=f_shape)
