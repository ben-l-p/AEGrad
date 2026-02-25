from __future__ import annotations
from copy import copy
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
    StructuralStates,
    StructuralDesignVariables,
)
from aegrad.algebra.se3 import hg_to_d
from algebra.se3 import exp_se3, t_se3

type StructuralObjectiveFunction = Callable[
    [StructuralStates, StructuralDesignVariables], Array
]


class BeamStructure(BaseBeamStructure):
    def _structural_states_res_from_dv_n(
        self,
        dv: StructuralDesignVariables,
        n: Array,
    ) -> tuple[StructuralStates, Array]:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        """

        # make a copy of the structure_dv object to prevent modifying the original states
        inner_case = BeamStructure(
            num_nodes=self.n_nodes,
            connectivity=self.connectivity,
            y_vector=self.y_vector,
            gravity=self.gravity_vec,
            verbosity=self.verbosity,
            optional_jacobians=self.optional_jacobians,
            convergence_settings=self.convergence_settings,
        )

        inner_case.set_design_variables(
            coords=dv.x0, k_cs=dv.k_cs, m_cs=dv.m_cs, m_lumped=dv.m_lumped
        )

        exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes_, 4, 4]
        hg = jnp.einsum("ijk,ikl->ijl", inner_case.hg0, exp_n)  # [n_nodes_, 4, 4]
        d = inner_case._make_d(hg)
        p_d = inner_case._make_p_d(d)
        eps = inner_case._make_eps(d)
        f_int = inner_case._assemble_vector_from_entries(
            inner_case._make_f_int(p_d, eps)
        ).reshape(-1, 6)
        if inner_case.use_gravity:
            m_t = inner_case._make_m_t(d)
            f_grav = inner_case._assemble_vector_from_entries(
                inner_case._make_f_grav(m_t, hg[:, :3, :3])
            ).reshape(-1, 6)
        else:
            m_t = None
            f_grav = None

        if dv.f_ext_dead is not None:
            f_ext_dead = inner_case._make_f_dead_ext(dv.f_ext_dead, hg[:, :3, :3])
        else:
            f_ext_dead = None

        ss = StructuralStates(
            hg=hg,
            d=d,
            eps=eps,
            f_int=f_int,
            f_ext_dead=f_ext_dead,
            f_ext_aero=None,
            f_grav=f_grav,
        )

        f_res = inner_case._make_f_res(
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

        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), structure.prescribed_dofs)
        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # base parameters
        n = vmap(hg_to_d)(self.hg0, structure.hg).ravel()  # [n_dof]
        t_n = vmap(t_se3, 0, 0)(n.reshape(-1, 6))  # [n_nodes_, 6, 6]
        d = structure.d
        eps = structure.eps
        p_d = self._make_p_d(d)
        m_t = self._make_m_t(d) if self.use_gravity else None

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

        struct_states = StructuralStates(
            hg=structure.hg,
            d=d,
            eps=eps,
            f_int=structure.f_int,
            f_ext_dead=structure.f_ext_dead,
            f_ext_aero=None,
            f_grav=structure.f_grav,
        )

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)
        n_u_full = self.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n = (jax.jacrev if n_f < n_u else jax.jacfwd)(
            lambda n_: objective(self._structural_states_res_from_dv_n(dv, n_)[0], dv)
        )(n).reshape(n_f, n_u_full)[:, solve_dofs]  # [n_f, n_u]

        # gradient of objective w.r.t. design variables
        p_f_p_x = (jax.jacrev if n_f < n_x else jax.jacfwd)(
            lambda dv_: objective(self._structural_states_res_from_dv_n(dv_, n)[0], dv_)
        )(dv).ravel_jacobian(n_f, n_x)  # [n_f, n_x]

        # gradient of residual w.r.t. minimal states
        # h and n have different tangent spaces, p_h_p_n = t(n)
        d_res_d_h = -self._make_k_t_full(
            d, p_d, eps, structure.f_ext_dead, structure.hg[:, :3, :3], m_t
        ).reshape(n_u_full, -1, 6)  # [n_u_full, n_nodes_, 6]
        d_res_d_n = jnp.einsum("ijk,jkl->ijl", d_res_d_h, t_n).reshape(
            n_u_full, n_u_full
        )[jnp.ix_(solve_dofs, solve_dofs)]

        # gradient of residual w.r.t. design variables
        p_res_p_x = (jax.jacrev if n_u < n_x else jax.jacfwd)(
            lambda dv_: self._structural_states_res_from_dv_n(dv_, n)[1]
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
