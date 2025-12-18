from jax import numpy as jnp
from jax import Array, vmap
from aegrad.utils import check_type
from aegrad.structure.structure_utils import check_connectivity
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.algebra.base import chi
from aegrad.structure.structure_utils import k_t_entry, g_int_entry
from aegrad.algebra.se3 import p
from typing import Optional
from functools import partial


class Structure:
    r"""
    Class to represent nonlinear beam structural model
    """

    def __init__(
        self,
        num_nodes: int,
        connectivity: Array,
        y_vector: Array,
        include_material: bool = True,
        include_geometric: bool = True,
    ) -> None:
        r"""
        Initialise Structure class with all non-design parameters
        :param num_nodes: Number of nodes in the structure
        :param connectivity: Connectivity array of shapes [n_elem, 2]
        :param y_vector: Vector defining the y direction for each element, [n_elem, 3]
        :param include_material: Whether to include material stiffness, defaults to True
        :param include_geometric: Whether to include geometric stiffness, defaults to True
        """

        self.include_material: bool = include_material
        self.include_geometric: bool = include_geometric
        if not self.include_material and not self.include_geometric:
            raise ValueError(
                "At least one of include_material or include_geometric must be True."
            )

        check_type(num_nodes, int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        check_arr_shape(connectivity, (None, 2), "connectivity")
        check_arr_dtype(connectivity, int, "connectivity")
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

        check_arr_shape(y_vector, (self.n_elem, 3), "y_vector")
        self.y_vector: Array = y_vector

        # initialize design variables with default values
        self.x0: Array = jnp.zeros((num_nodes, 3))
        self.m: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k: Array = jnp.zeros((self.n_elem, 6, 6))
        self.m_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_cs: Array = jnp.zeros((self.n_elem, 6, 6))

        # initialise auxiliary arrays
        self.o0: Array = jnp.zeros((self.n_elem, 3, 3))
        self.l0: Array = jnp.zeros(self.n_elem)
        self.d0: Array = jnp.zeros((self.n_elem, 6))

    def set_design_variables(
        self, coords: Array, k_cs: Array, m_cs: Optional[Array]
    ) -> None:
        r"""
        Set design variables and compute initial configuration dependent quantities
        :param coords: Node coordinates, [n_nodes, 3]
        :param k_cs: Cross-section stiffness matrices, [n_elem, 6, 6]
        :param m_cs: Cross-section mass matrices, [n_elem, 6, 6]
        """
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
        dz = jnp.cross(dx_unit, self.y_vector, axis=-1)  # vector in plane[n_elem, 3]
        dz_unit = dz / jnp.linalg.norm(dz, axis=-1)[:, None]  # [n_elem, 3]

        dy_unit = jnp.cross(dz_unit, dx_unit)

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

    def make_k(
        self,
        d: Array,
        p_d: Array,
        eps: Array,
    ) -> Array:
        r"""
        Assemble global stiffness matrix as a function of the element relative configuration vectors
        :param d: Element relative configuration, [n_elem, 6]
        :return: Global stiffness matrix, [n_dof, n_dof]
        """
        # compute stiffness matrix entries
        k_t_entries = vmap(
            partial(
                k_t_entry,
                include_material=self.include_material,
                include_geometric=self.include_geometric,
            ),
            (0, 0, 0, 0, 0),
            0,
        )(d, p_d, self.l0, eps, self.k)  # [n_elem, 12, 12]

        # assemble global stiffness matrix
        k_t = jnp.zeros((self.n_dof, self.n_dof))
        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            k_t = k_t.at[jnp.ix_(index, index)].set(k_t_entries[i_elem, ...])

        return k_t

    def make_g_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param d: Element relative configuration, [n_elem, 6]
        :return: Global internal force vector, [n_dof]
        """

        # compute internal force entries
        g_int_entries = vmap(g_int_entry, (0, 0, 0), 0)(
            p_d, self.k, eps
        )  # [n_elem, 12]

        # assemble global internal force vector
        g_int = jnp.zeros(self.n_dof)
        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            g_int = g_int.at[index].add(g_int_entries[i_elem, ...])

        return g_int

    def make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Element strain vectors, [n_elem, 6]
        """

        return (d - self.d0) / self.l0[:, None]

    def make_g_int_and_k_t(self, d: Array) -> tuple[Array, Array]:
        r"""
        Compute both the internal force vector and tangent stiffness matrix as a function of the element relative
        configuration vectors. This makes computation more efficient by reusing intermediate results.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Tuple of global internal force vector [n_dof] and global stiffness matrix [n_dof, n_dof]
        """

        eps = self.make_eps(d)  # [n_elem, 6]

        # compute P(d) matrices
        p_d = vmap(p)(d)  # [n_elem, 6, 12]

        g_int = self.make_g_int(p_d, eps)  # [n_dof]
        k_t = self.make_k(d, p_d, eps)  # [n_dof, n_dof]

        return g_int, k_t

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

        # def newton_raphson(
        #     func: Callable[[Array], Array],
        #     jac: Callable[[Array], Array],
        #     x_init: Array,
        #     free_dof: slice,
        # ) -> Array:
        #     n_iter = 10
        #
        #     def update(_, x_km1: Array) -> Array:
        #         f = func(x_km1)[free_dof]  # [m - n_cnst]
        #         j = jac(x_km1)[free_dof, free_dof]  # [m - n_cnst, m - n_cnst]
        #         dx = jnp.linalg.solve(j, f)  # [m - n_cnst]
        #         return x_km1.at[free_dof].add(-dx)
        #
        #     return jax.lax.fori_loop(0, n_iter, update, x_init)
        #     # return update(0, x_init)
        #
        # return newton_raphson(g_ext)
