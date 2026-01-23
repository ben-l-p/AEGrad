import jax.lax
from jax import numpy as jnp
from jax import Array, vmap
from aegrad.utils import check_type
from aegrad.structure.structure_utils import check_connectivity, n_elem_per_node
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.structure.structure_utils import k_t_entry
from aegrad.algebra.se3 import p, ha_to_d, log_se3, rmat_to_ha_hat
from typing import Optional, Sequence
from functools import partial
from aegrad.algebra.solvers import newton_raphson


class Structure:
    r"""
    Class to represent nonlinear beam structural model
    """

    def __init__(
        self,
        num_nodes: int,
        connectivity: Array,
        y_vector: Array,
    ) -> None:
        r"""
        Initialise Structure class with all non-design parameters
        :param num_nodes: Number of nodes in the structure
        :param connectivity: Connectivity array of shapes [n_elem, 2]
        :param y_vector: Vector defining the y direction for each element, [n_elem, 3]
        """

        check_type(num_nodes, int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        check_arr_shape(connectivity, (None, 2), "connectivity")
        check_arr_dtype(connectivity, int, "connectivity")
        check_connectivity(connectivity, num_nodes)
        self.connectivity: Array = connectivity  # [n_elem, 2]
        self.n_elem_per_node: Array = n_elem_per_node(connectivity)  # [n_nodes]
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
        self.m_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_l: Array = jnp.zeros((self.n_elem, 6, 6))

        # initialise auxiliary arrays
        self.o0: Array = jnp.zeros((self.n_elem, 3, 3))
        self.l0: Array = jnp.zeros(self.n_elem)
        self.d0: Array = jnp.zeros((self.n_elem, 6))

        # initialise undeformed algebra and group
        self.hg0: Array = jnp.zeros((self.n_nodes, 4, 4))
        self.ha0: Array = jnp.zeros((self.n_nodes, 6))

        # adjoint inverse action for the reference rotations
        self.ad_inv_o0: Array = jnp.zeros((self.n_elem, 6, 6))

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

        # ensure out-of-plane vector and beam vector are not collinear
        if jnp.any(jnp.linalg.norm(jnp.cross(dx, self.y_vector, 1, 1), axis=-1) < 1e-6):
            raise ValueError(
                "y_vector is collinear with beam element direction for at least one element. "
                "Please provide a different y_vector."
            )

        self.l0 = self.l0.at[...].set(jnp.linalg.norm(dx, axis=-1))  # [n_elem]
        self.d0 = self.d0.at[:, 0].set(self.l0)

        dx_unit = dx / self.l0[:, None]  # unit vector in beam direction, [n_elem, 3]
        dz = jnp.cross(dx_unit, self.y_vector, axis=-1)  # vector in plane[n_elem, 3]
        dz_unit = dz / jnp.linalg.norm(dz, axis=-1)[:, None]  # [n_elem, 3]

        dy_unit = jnp.cross(dz_unit, dx_unit)

        self.o0 = self.o0.at[..., 0].set(dx_unit)
        self.o0 = self.o0.at[..., 1].set(dy_unit)
        self.o0 = self.o0.at[..., 2].set(dz_unit)

        self.ad_inv_o0 = self.ad_inv_o0.at[...].set(
            vmap(rmat_to_ha_hat)(jnp.transpose(self.o0, (0, 2, 1)))
        )

        # self.ad_inv_o0 = self.ad_inv_o0.at[...].set(vmap(rmat_to_ha_hat)(self.o0))

        # self.ad_inv_o0 = self.ad_inv_o0.at[...].set(jnp.eye(6)[None, ...])

        self.hg0 = jnp.broadcast_to(
            jnp.eye(4)[None, ...], (self.n_nodes, 4, 4)
        )  # [n_nodes, 4, 4]
        self.hg0 = self.hg0.at[:, :3, 3].set(self.x0)

        self.ha0 = vmap(log_se3)(self.hg0)  # [n_nodes, 6]

        self.k_l = self.k_l.at[...].set(self.k_cs * self.l0[:, None, None])

    def make_k(
        self,
        d: Array,
        p_d: Array,
        eps: Array,
        include_material: bool,
        include_geometric: bool,
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
                include_material=include_material,
                include_geometric=include_geometric,
            ),
            (0, 0, 0, 0, 0, 0, 0),
            0,
        )(
            d,
            p_d,
            self.l0,
            eps,
            self.k_cs,
            self.ad_inv_o0[self.connectivity[:, 0], ...],
            self.ad_inv_o0[self.connectivity[:, 1], ...],
        )  # [n_elem, 12, 12]

        # assemble global stiffness matrix
        k_t = jnp.zeros((self.n_dof, self.n_dof))
        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            k_t = k_t.at[jnp.ix_(index, index)].add(k_t_entries[i_elem, ...])

        return k_t

    def make_g_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Global internal force vector, [n_dof]
        """

        g_int_entries = jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs, eps)

        # assemble global internal force vector
        g_int = jnp.zeros(self.n_dof)
        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            g_int = g_int.at[index[:6]].add(g_int_entries[i_elem, :6])
            g_int = g_int.at[index[6:]].add(g_int_entries[i_elem, 6:])

        return g_int

    def make_g_ext_ab(self, g_ext: Array, d: Array) -> Array:
        r"""
        Compute the global external force vector as a function of the element relative configuration vectors
        :param g_ext: External forces array of follower forces [n_elem, 2, 6]
        :param d: Element relative configuration, [n_elem, 6]
        :return: Global external force vector, [n_dof]
        """

        # rots = jnp.transpose(
        #     self.o0[self.connectivity, :, :], (0, 1, 3, 2)
        # )  # [n_elem, 2, 3, 3]
        # chis = vmap(vmap(chi, 0, 0), 1, 1)(rots)  # [n_elem, 2, 6, 6]
        # g_ext_rot = jnp.einsum("ijkl,ijl->ijk", chis, g_ext)  # [n_elem, 2, 6]

        # g_ext_ab_0 = jnp.einsum(
        #     "ikj,ik->ij",
        #     self.ad_inv_o0,
        #     g_ext_rot[:, 0, :],
        # )  # [n_elem, 6]
        #
        # g_ext_ab_l = jnp.einsum(
        #     "ikj,ik->ij",
        #     self.ad_inv_o0,
        #     g_ext_rot[:, 1, :],
        # )  # [n_elem, 6]

        g_ext_ab_0 = jnp.einsum(
            "ikj,ik->ij",
            self.ad_inv_o0,
            g_ext[:, 0, :],
        )  # [n_elem, 6]

        g_ext_ab_l = jnp.einsum(
            "ikj,ik->ij",
            self.ad_inv_o0,
            g_ext[:, 1, :],
        )  # [n_elem, 6]

        g_ext_ab_vec = jnp.zeros(self.n_dof)

        for i_elem in range(self.n_elem):
            index = self.dof_per_elem[i_elem, :]
            g_ext_ab_vec = g_ext_ab_vec.at[index[:6]].add(g_ext_ab_0[i_elem, ...])
            g_ext_ab_vec = g_ext_ab_vec.at[index[6:]].add(g_ext_ab_l[i_elem, ...])

        return g_ext_ab_vec

    def make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Element strain vectors, [n_elem, 6]
        """

        return (d - self.d0) / self.l0[:, None]

    def make_g_int_and_k_t(
        self, d: Array, include_material: bool, include_geometric: bool
    ) -> tuple[Array, Array]:
        r"""
        Compute both the internal force vector and tangent stiffness matrix as a function of the element relative
        configuration vectors. This makes computation more efficient by reusing intermediate results.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Tuple of global internal force vector [n_dof] and global stiffness matrix [n_dof, n_dof]
        """

        eps = self.make_eps(d)  # [n_elem, 6]

        # compute P(d) matrices
        p_d = vmap(p, (0, 0, 0), 0)(
            d,
            self.ad_inv_o0[self.connectivity[:, 0], ...],
            self.ad_inv_o0[self.connectivity[:, 1], ...],
        )  # [n_elem, 6, 12]

        g_int = self.make_g_int(p_d, eps)  # [n_dof]
        k_t = self.make_k(
            d, p_d, eps, include_material, include_geometric
        )  # [n_dof, n_dof]

        return g_int, k_t

    def static_solve(
        self,
        g_ext: Array,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        ha_init: Optional[Array] = None,
        include_material: bool = True,
        include_geometric: bool = False,
        load_steps: int = 1,
        max_iter: int = 40,
    ) -> tuple[Array, Array, Array]:
        r"""
        Perform static solve of the structure under external loads
        :param g_ext: External forces array of follower forces [n_elem, 2, 6]
        :param prescribed_dofs: Array of prescribed dof indices
        :param ha_init: Initial guess for the algebra, [n_nodes, 6]
        :return: Node algebra [n_nodes, 6], configuration vectors [n_elem, 6] and internal forces, [n_nodes, 6]
        """

        # degrees of freedom which are prescribed
        if isinstance(prescribed_dofs, slice) or isinstance(prescribed_dofs, Sequence):
            prescribed_dofs_arr = jnp.arange(self.n_dof)[prescribed_dofs]
        elif isinstance(prescribed_dofs, int):
            prescribed_dofs_arr = jnp.array([prescribed_dofs])
        elif isinstance(prescribed_dofs, Array):
            prescribed_dofs_arr = prescribed_dofs
        else:
            raise TypeError(
                "prescribed_dofs must be an int, slice, Sequence[int], or Array"
            )
        if load_steps < 1:
            raise ValueError("load_steps must be at least 1")

        # set initial states
        if ha_init is None:
            ha0 = self.ha0.ravel()
        else:
            ha0 = ha_init.ravel()

        # degrees of freedom to solve for
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)
        n_solve_dofs = solve_dofs.shape[0]

        def ha_solve_to_full(ha_solve: Array) -> Array:
            r"""
            Reconstruct full algebra from solved degrees of freedom
            :param ha_solve: Solved degrees of freedom, [n_solve_dof]
            :return: Full algebra, [n_dof]
            """
            ha_full = jnp.zeros(self.n_dof)
            ha_full = ha_full.at[solve_dofs].set(ha_solve)
            return ha_full.at[prescribed_dofs_arr].set(ha0[prescribed_dofs_arr])

        def residual_and_jac_func(ha_solve: Array, g_ext: Array) -> tuple[Array, Array]:
            r"""
            Compute the residual and Jacobian for the Newton-Raphson solver.
            :param ha_solve: Solved degrees of freedom, [n_solve_dof]
            :param g_ext: External forces array of follower forces [n_elem, 2, 6]
            :return:
            """
            ha = ha_solve_to_full(ha_solve).reshape(self.n_nodes, 6)
            d = vmap(ha_to_d, (0, 0), 0)(
                ha[self.connectivity[:, 0], :], ha[self.connectivity[:, 1], :]
            )  # [n_elem, 6]
            g_int, k_t = self.make_g_int_and_k_t(
                d, include_material, include_geometric
            )  # [n_dof], [n_dof, n_dof]
            g_ext_ab = self.make_g_ext_ab(g_ext, d)  # [n_dof]
            g_res = g_int - g_ext_ab  # [n_dof]
            return g_res[solve_dofs], k_t[jnp.ix_(solve_dofs, solve_dofs)]

        def load_step_func(
            i_step: int,
            carry: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            g_ext_step = g_ext * (i_step + 1) / load_steps
            return newton_raphson(
                lambda ha_: residual_and_jac_func(ha_, g_ext_step),
                carry[0],  # previous step solution as initial guess
                atol=1e-10,
                rtol=1e-7,
                n_iter_max=max_iter,
            )

        ha_solve, g_res_solve, k_t_solve = jax.lax.fori_loop(
            0,
            load_steps,
            load_step_func,
            (
                ha0[solve_dofs],
                jnp.zeros(n_solve_dofs),
                jnp.zeros((n_solve_dofs, n_solve_dofs)),
            ),
        )

        # reconstruct full algebra, [n_dof]
        ha_full = ha_solve_to_full(ha_solve).reshape(self.n_nodes, 6)
        d_full = vmap(ha_to_d, (0, 0), 0)(
            ha_full[self.connectivity[:, 0]], ha_full[self.connectivity[:, 1]]
        )  # [n_elem, 6]
        g_int_full = self.make_g_int_and_k_t(
            d_full, include_material, include_geometric
        )[0].reshape(self.n_nodes, 6)

        # hg_full =

        return (
            ha_full,
            d_full,
            g_int_full,
        )
