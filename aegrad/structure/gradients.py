from __future__ import annotations
from copy import copy
from typing import Optional, Callable, Sequence
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax import Array, vmap

from aegrad.structure.beam import BeamStructure
from aegrad.structure.data_structures import StaticStructure
from aegrad.algebra.se3 import hg_to_d
from algebra.se3 import exp_se3
from aegrad.utils import _make_pytree
from aegrad.algebra.array_utils import check_arr_shape

type ObjectiveFunction = Callable[[StructuralStates, DesignVariables], Array]


@_make_pytree
class DesignVariables:
    def __init__(
        self,
        coords: Optional[Array],
        k_cs: Optional[Array],
        m_cs: Optional[Array],
        m_lumped: Optional[Array],
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
    ):
        self.coords: Array = coords
        self.k_cs: Array = k_cs
        self.m_cs: Optional[Array] = m_cs
        self.m_lumped: Optional[Array] = m_lumped
        self.f_ext_follower: Optional[Array] = f_ext_follower
        self.f_ext_dead: Optional[Array] = f_ext_dead

        self.shapes: dict[str, Optional[tuple[int, ...]]] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def get_vars(self) -> dict[str, Optional[Array]]:
        return {
            "coords": self.coords,
            "k_cs": self.k_cs,
            "m_cs": self.m_cs,
            "m_lumped": self.m_lumped,
            "f_ext_follower": self.f_ext_follower,
            "f_ext_dead": self.f_ext_dead,
        }

    def get_shapes(self) -> dict[str, Optional[tuple[int, ...]]]:
        return {
            k: var.shape if var is not None else None
            for k, var in self.get_vars().items()
        }

    def make_index_mapping(self) -> tuple[dict[str, Optional[Array]], int]:
        mapping = {}
        cnt = 0
        for name, shape in self.shapes.items():
            if shape is not None:
                var_size = jnp.prod(jnp.array(shape))
                mapping[name] = jnp.arange(cnt, cnt + var_size).reshape(shape)
                cnt += var_size
            else:
                mapping[name] = None
        return mapping, cnt

    def ravel_jacobian(self, f_size: int, x_size: int) -> Array:
        arr = jnp.concatenate(
            [
                var.reshape(f_size, -1)
                for var in self.get_vars().values()
                if var is not None
            ],
            axis=1,
        )
        check_arr_shape(arr, (f_size, x_size), "Internal jacobian")
        return arr

    def ravel(self) -> Array:
        return jnp.concatenate(
            [var.ravel() for var in self.get_vars().values() if var is not None]
        )

    def reshape(self, *args: int) -> Array:
        return self.ravel().reshape(*args)

    def from_adjoint(self, f_shape: tuple[int, ...], df_dx: Array) -> DesignVariables:
        out_dict = {}
        for name in self.shapes.keys():
            if self.mapping[name] is not None:
                out_dict[name] = df_dx[:, self.mapping[name]].reshape(
                    *f_shape, *self.shapes[name]
                )
            else:
                out_dict[name] = None
        return DesignVariables(**out_dict)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ("shapes", "mapping")

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return ("coords", "k_cs", "m_cs", "m_lumped", "f_ext_follower", "f_ext_dead")


@jax.tree_util.register_dataclass
@dataclass
class StructuralStates:
    hg: Array
    d: Array
    eps: Array
    f_int: Array
    f_ext_dead: Optional[Array]
    f_grav: Optional[Array]


class BeamGradients(BeamStructure):
    def solution_from_n(
        self, n: Array, f_ext_dead: Optional[Array]
    ) -> StructuralStates:
        r"""
        Use the derivative of this for stuff.
        """

        exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes, 4, 4]
        hg = jnp.einsum("ijk,ikl->ijl", self.hg0, exp_n)  # [n_nodes, 4, 4]
        d = self._make_d(hg)
        p_d = self._make_p_d(d)
        eps = self._make_eps(d)
        f_int = self._make_f_int(p_d, eps)
        if self.use_gravity:
            m_t = self._make_m_t(d)
            f_grav = self._make_f_grav(m_t, hg[:, :3, :3])
        else:
            f_grav = None

        if f_ext_dead is not None:
            f_ext_dead = self._make_f_dead_ext(f_ext_dead, hg[:, :3, :3])

        return StructuralStates(
            hg=hg, d=d, eps=eps, f_int=f_int, f_ext_dead=f_ext_dead, f_grav=f_grav
        )

    def test_n(self, n: Array, f_ext_dead: Optional[Array] = None) -> Array:
        r"""
        Use the derivative of this for stuff.
        """

        exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes, 4, 4]
        hg = jnp.einsum("ijk,ikl->ijl", self.hg0, exp_n)  # [n_nodes, 4, 4]
        return self._make_d(hg)

    def static_adjoint(
        self,
        structure: StaticStructure,
        objective: ObjectiveFunction,
        prescribed_dofs: Sequence[int] | Array | slice | int,
    ) -> DesignVariables:
        r"""
        Computes the static adjoint of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param structure: StaticStructure containing the current state of the structure.
        :param objective: Objective function that takes the structure and design variables and returns an array
        :return: Adjoint variables.
        """

        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)

        n = vmap(hg_to_d)(self.hg0, structure.hg).ravel()  # [n_dof]

        d = structure.d
        eps = structure.eps
        p_d = self._make_p_d(d)
        m_t = self._make_m_t(d) if self.use_gravity else None

        # gradient of residual w.r.t. states
        dr_du = self._make_k_t_full(
            d, p_d, eps, structure.f_ext_dead, structure.hg[:, :3, :3], m_t
        )[jnp.ix_(solve_dofs, solve_dofs)]

        gs = copy(structure)
        gs.to_global()
        dv = DesignVariables(
            coords=self.x0,
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
            f_grav=structure.f_grav,
        )

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)

        # gradient of objective w.r.t. states
        pf_pu = (jax.jacrev if n_f < n_u else jax.jacfwd)(
            lambda n_: objective(self.solution_from_n(n_, dv.f_ext_dead), dv)
        )(n)[:, solve_dofs]  # [n_f, n_u]

        # gradient of object w.r.t. design variables
        pf_px = (jax.jacrev if n_f < n_x else jax.jacfwd)(
            lambda dv_: objective(struct_states, dv_)
        )(dv).ravel_jacobian(n_f, n_x)  # [n_f, n_x]

        def f_res_from_dv(dv_: DesignVariables) -> Array:
            inner_case = BeamStructure(
                self.n_nodes,
                self.connectivity,
                self.y_vector,
                self.gravity_vec,
                self.include_geometric,
                self.include_q_dot,
            )

            inner_case.set_design_variables(
                coords=dv_.coords, k_cs=dv_.k_cs, m_cs=dv_.m_cs, m_lumped=dv_.m_lumped
            )

            exp_n = vmap(exp_se3)(n.reshape(-1, 6))  # [n_nodes, 4, 4]
            hg = jnp.einsum("ijk,ikl->ijl", inner_case.hg0, exp_n)  # [n_nodes, 4, 4]
            d = inner_case._make_d(hg)
            p_d = inner_case._make_p_d(d)
            eps = inner_case._make_eps(d)
            m_t = inner_case._make_m_t(d) if inner_case.use_gravity else None

            return inner_case._make_f_res(
                solve_dofs=None,
                p_d=p_d,
                eps=eps,
                hg=hg,
                f_ext_follower_n=dv_.f_ext_follower,
                f_ext_dead_n=dv_.f_ext_dead,
                dynamic=False,
                m_t=m_t,
                c_l=None,
                c_l_lumped=None,
                v=None,
                v_dot=None,
            )[0][solve_dofs]

        # gradient of residual w.r.t. design variables
        pr_px = (jax.jacrev if n_u < n_x else jax.jacfwd)(f_res_from_dv)(
            dv
        ).ravel_jacobian(n_u, n_x)  # [n_u, n_x]

        if n_f < n_x:
            # reverse mode
            adjoint = jnp.linalg.solve(dr_du, pf_pu.T).T  # [n_f, n_u]

            rhs = adjoint @ pr_px  # [n_f, n_x]
        else:
            # forward mode
            adjoint = jnp.linalg.solve(dr_du, pr_px).reshape(
                self.n_nodes, 6, n_x
            )  # [n_nodes*6, n_x]

            rhs = jnp.einsum("ij,ijk->k", pf_pu, adjoint)  # [n_f, n_x]

        return dv.from_adjoint(f_shape, pf_px + rhs)
