from jax import numpy as jnp
from jax import Array, vmap
from aegrad.utils import check_type
from aegrad.structure.utils import check_connectivity
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.algebra.base import chi
from aegrad.algebra.se3 import p, k_tg_entry
from typing import Optional


class Structure:
    r"""
    Class to represent structural model
    """

    def __init__(self, num_nodes: int, connectivity: Array, normal_vector: Array):
        r"""
        Initialize Structure class with all non-design parameters
        :param num_nodes: Number of nodes in the structure
        :param connectivity: Connectivity array of shapes [n_elem, 2]
        """
        check_type(num_nodes, int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        check_arr_shape(connectivity, (None, 2))
        check_arr_dtype(connectivity, int)
        check_connectivity(connectivity, num_nodes)
        self.connectivity: Array = connectivity  # [n_elem, 2]
        self.n_elem: int = connectivity.shape[0]

        self.dof_per_elem: Array = jnp.zeros((self.n_elem, 12), dtype=int)
        self.dof_per_elem = self.dof_per_elem.at[:, :6].set(
            6 * self.connectivity[:, [0]] + jnp.arange(6)[None, :]
        )
        self.dof_per_elem = self.dof_per_elem.at[:, 6:].set(
            6 * self.connectivity[:, [1]] + jnp.arange(6)[None, :]
        )

        check_arr_shape(normal_vector, (self.n_elem, 3))
        self.plane_vector: Array = normal_vector

        # initialize design variables with default values
        self.x0: Array = jnp.zeros((num_nodes, 3))
        self.m: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k: Array = jnp.zeros((self.n_elem, 6, 6))
        self.m_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_cs: Array = jnp.zeros((self.n_elem, 6, 6))

        # initialise auxillary arrays
        self.o0: Array = jnp.zeros((self.n_elem, 3, 3))
        self.l0: Array = jnp.zeros(self.n_elem)
        self.d0: Array = jnp.zeros((self.n_elem, 6))

    def set_design_variables(
        self, coords: Array, k_cs: Array, m_cs: Optional[Array]
    ) -> None:
        # populate arrays
        self.k_cs = self.k_cs.at[...].set(k_cs)
        if m_cs is not None:
            self.m_cs = self.m_cs.at[...].set(m_cs)
        self.x0 = self.x0.at[...].set(coords)

        # obtain initial orientation and length
        x_elem = jnp.take(self.x0, self.connectivity, axis=0)  # [n_elem, 2, 3]
        dx = x_elem[:, 1, :] - x_elem[:, 0, :]  # [n_elem, 3]
        self.l0 = self.l0.at[...].set(jnp.linalg.norm(dx, axis=-1))  # [n_elem]
        self.d0 = self.d0.at[:, 0].set(self.l0)

        dx_unit = dx / self.l0[:, None]  # unit vector in beam direction, [n_elem, 3]
        dy = jnp.cross(
            dx_unit, self.plane_vector, axis=-1
        )  # vector in plane[n_elem, 3]
        dy_unit = dy / jnp.linalg.norm(dy, axis=-1)[:, None]  # [n_elem, 3]
        dz_unit = jnp.cross(
            dx_unit, dy_unit, axis=-1
        )  # unit vector out of plane, [n_elem, 3]

        self.o0 = self.o0.at[..., 0].set(dx_unit)
        self.o0 = self.o0.at[..., 1].set(dy_unit)
        self.o0 = self.o0.at[..., 2].set(dz_unit)

        chi0 = vmap(chi)(self.o0)  # [n_elem, 6, 6]
        self.k = self.k.at[...].set(
            jnp.einsum("ijk,ikl,iml->ijm", chi0, self.k_cs, chi0)
        )
        self.m = self.m.at[...].set(
            jnp.einsum("ijk,ikl,iml->ijm", chi0, self.m_cs, chi0)
        )

    def make_k_t(self, d: Array, include_geometric: bool = True) -> Array:
        pd = vmap(p, 0, 0)(d)  # [n_elem, 6, 12]
        pd_l = pd / self.l0[:, None, None]  # [n_elem, 6, 12]
        k_t_entries = jnp.einsum(
            "ikj,ikl,ilm->ijm", pd, self.k, pd_l
        )  # [n_elem, 12, 12]

        if include_geometric:
            eps = (d - self.d0) / self.l0[:, None]
            k_t_entries += vmap(k_tg_entry, (0, 0, 0), 0)(
                d, eps, self.k
            )  # [n_elem, 12, 12]

        k_t = jnp.zeros((self.n_dof, self.n_dof))
        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            k_t = k_t.at[jnp.ix_(index, index)].set(k_t_entries[i_elem, ...])

        return k_t

    def static_solve(
        self, g_ext: Array, prescribed_dofs: Array, prescribed_values: Array
    ) -> Array:
        r"""
        Perform static solve of the structure under external loads
        :param g_ext: External forces array of shapes [n_nodes, 6]
        :param prescribed_dofs: Array of prescribed dof indices
        :param prescribed_values: Array of prescribed dof values
        :return: Displacement array of shapes [n_nodes, 6]
        """
        pass
