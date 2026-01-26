from jax import numpy as jnp
from jax import Array, vmap
import jax
import equinox
from aegrad.utils import check_type
from aegrad.structure.data_structures import StaticStructure
from aegrad.structure.structure_utils import check_connectivity, n_elem_per_node
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.structure.structure_utils import k_t_entry, integrate_m_l
from aegrad.algebra.se3 import p, rmat_to_ha_hat, hg_to_d, exp_se3
from typing import Optional, Sequence
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

        self.hg0 = jnp.broadcast_to(
            jnp.eye(4)[None, ...], (self.n_nodes, 4, 4)
        )  # [n_nodes, 4, 4]
        self.hg0 = self.hg0.at[:, :3, 3].set(self.x0)

        self.k_l = self.k_l.at[...].set(self.k_cs * self.l0[:, None, None])

    def assemble_matrix_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global matrix from element entries
        :param entries: Array of element matrix entries, [n_elem, 12, 12]
        :return: System global matrix, [n_dof, n_dof]
        """

        # vectorised assembly of full matrix
        row_idx = jnp.broadcast_to(self.dof_per_elem[:, :, None], (self.n_elem, 12, 12))
        col_idx = jnp.broadcast_to(self.dof_per_elem[:, None, :], (self.n_elem, 12, 12))
        return (
            jnp.zeros((self.n_dof, self.n_dof))
            .at[row_idx.ravel(), col_idx.ravel()]
            .add(entries.ravel())
        )

    def make_k_t(
        self,
        d: Array,
        p_d: Array,
        eps: Array,
        include_material: bool,
        include_geometric: bool,
    ) -> Array:
        r"""
        Assemble tangent stiffness matrix as a function of the element relative configuration vectors
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
            (0, 0, 0, 0, 0, 0),
            0,
        )(
            d,
            p_d,
            self.l0,
            eps,
            self.k_cs,
            self.ad_inv_o0,
        )  # [n_elem, 12, 12]

        return self.assemble_matrix_from_entries(k_t_entries)

    def make_m_t(self, d: Array, int_order: int = 3) -> Array:
        r"""
        Assemble tangent mass matrix as a function of the element relative configuration vectors
        :param d: Element relative configuration, [n_elem, 6]
        :param int_order: Integration order for mass matrix computation
        :return: Global mass matrix, [n_dof, n_dof]
        """
        m_t_entries = vmap(
            partial(integrate_m_l, int_order=int_order), (0, 0, 0, 0), 0
        )(d, self.m_cs, self.ad_inv_o0, self.l0)

        return self.assemble_matrix_from_entries(m_t_entries)

    def make_f_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Global internal force vector, [n_dof]
        """

        g_int_entries = jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs, eps)

        # assemble global internal force vector
        g_int = jnp.zeros(self.n_dof)
        g_int = g_int.at[self.dof_per_elem[:, :6]].add(g_int_entries[:, :6])
        g_int = g_int.at[self.dof_per_elem[:, 6:]].add(g_int_entries[:, 6:])

        return g_int

    @staticmethod
    def make_f_dead_ext_ab(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, [n_node, 6]
        :param rmat: Deformation rotation matrices, [n_node, 3, 3]
        :return: Global external force vector, [n_dof]
        """

        # f_rot = jnp.einsum("ijk,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        # m_rot = jnp.einsum("ijk,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        f_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        m_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        return jnp.concatenate((f_rot, m_rot), axis=-1).ravel()

    def make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Element strain vectors, [n_elem, 6]
        """

        return (d - self.d0) / self.l0[:, None]

    def make_f_int_and_k_t(
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

        f_int = self.make_f_int(p_d, eps)  # [n_dof]
        k_t = self.make_k_t(
            d, p_d, eps, include_material, include_geometric
        )  # [n_dof, n_dof]

        return f_int, k_t

    def make_d(self, hg: Array) -> Array:
        r"""
        Compute the element relative configuration vectors from the nodal homogeneous transformation matrices
        :param hg: Nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        :return: Element relative configuration vectors, [n_elem, 6]
        """

        base_hg = jnp.zeros((self.n_elem, 4, 4))
        base_hg = base_hg.at[:, :3, :3].set(self.o0)
        base_hg = base_hg.at[:, 3, 3].set(1.0)

        haha0 = jnp.einsum(
            "ijk,ikl->ijl", hg[self.connectivity[:, 0], :, :], base_hg
        )  # [n_elem, 4, 4]
        haha1 = jnp.einsum(
            "ijk,ikl->ijl", hg[self.connectivity[:, 1], :, :], base_hg
        )  # [n_elem, 4, 4]

        return vmap(hg_to_d, (0, 0), 0)(haha0, haha1)  # [n_elem, 6]

    def static_solve(
        self,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        prescribed_dofs: Sequence[int] | Array | slice | int,
        include_material: bool = True,
        include_geometric: bool = False,
        load_steps: int = 1,
        max_iter: int = 40,
        rel_tol: float = 1e-7,
        abs_tol: float = 1e-9,
    ) -> StaticStructure:
        r"""
        Perform static solve of the structure under external loads
        :param f_ext_follower: External forces array of follower forces [n_node, 6]
        :param f_ext_dead: External forces array of dead loads [n_node, 6]
        :param prescribed_dofs: Index of degrees of freedom which are prescribed (not solved for).
        :param include_material: Whether to include material stiffness contribution.
        :param include_geometric: Whether to include geometric stiffness contribution.
        :param load_steps: Number of load steps to apply the external loads over.
        :param max_iter: Maximum number of Newton-Raphson iterations per load step.
        :param rel_tol: Relative tolerance for convergence. measured for resultant forces relative to maximum applied
        force.
        :param abs_tol: Absolute tolerance for convergence.
        :return: StaticStructure dataclass containing results of the static analysis.
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

        # check inputs
        if f_ext_follower is not None:
            check_arr_shape(f_ext_follower, (self.n_nodes, 6), "f_ext_follower")
        if f_ext_dead is not None:
            check_arr_shape(f_ext_dead, (self.n_nodes, 6), "f_ext_dead")

        # degrees of freedom to solve for
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)
        n_solve_dofs = solve_dofs.shape[0]

        def update(
            hg_n_full: Array,
            f_res_n: Array,
            f_ext_follower_n: Array,
            f_ext_dead_n: Array,
        ) -> tuple[Array, Array, Array, Array]:
            # configuration vectors, [n_elem, 6]
            d_n = self.make_d(hg_n_full)

            f_int_n, k_t_n = self.make_f_int_and_k_t(
                d_n, include_material, include_geometric
            )  # [n_dof], [n_dof, n_dof]

            k_t_n_solve = k_t_n[
                jnp.ix_(solve_dofs, solve_dofs)
            ]  # [n_solve_dofs, n_solve_dofs]
            g_int_n_solve = f_int_n[solve_dofs]  # [n_solve_dofs]

            f_res_n_solve = g_int_n_solve
            if f_ext_follower is not None:
                f_res_n_solve -= f_ext_follower_n.reshape(self.n_dof).ravel()[
                    solve_dofs
                ]
            if f_ext_dead is not None:
                f_res_n_solve -= self.make_f_dead_ext_ab(
                    f_ext_dead_n, hg_n_full[:, :3, :3]
                )[solve_dofs]

            d_ha_np1 = -jnp.linalg.solve(k_t_n_solve, f_res_n_solve)
            d_ha_np1_full = jnp.zeros(self.n_dof)
            d_ha_np1_full = d_ha_np1_full.at[solve_dofs].set(d_ha_np1)

            hg_np1_full = jnp.einsum(
                "ijk,ikl->ijl",
                hg_n_full,
                vmap(exp_se3, 0, 0)(d_ha_np1_full.reshape(-1, 6)),
            )
            return hg_np1_full, f_res_n, f_ext_follower_n, f_ext_dead_n

        def convergence(
            hg_n_full: Array,
            g_res_n: Array,
            f_ext_follower_n: Array,
            f_ext_dead_n: Array,
        ) -> Array:
            max_res = jnp.abs(g_res_n).max()

            # find maximum external force magnitude from both forcing types
            max_ext = jnp.zeros(())
            if f_ext_follower_n is not None:
                max_ext = jnp.abs(f_ext_follower_n).max()
            if f_ext_dead_n is not None:
                max_ext = jnp.maximum(max_ext, jnp.abs(f_ext_dead_n).max())

            conv_abs = max_res < abs_tol

            # rely on abs_tol if external forces are very small
            conv_rel = jax.lax.select(
                max_ext > 1e-5, (max_res / max_ext) < rel_tol, jnp.zeros((), dtype=bool)
            )

            # return true if converged
            return conv_abs | conv_rel

        def inner_loop(
            f_ext_follower_n: Array,
            f_ext_dead_n: Array,
            g_res_init: Array,
            hg_init: Array,
        ) -> tuple[Array, Array, Array, Array]:
            return equinox.internal.while_loop(
                lambda args: convergence(*args),
                lambda args: update(*args),
                (hg_init, g_res_init, f_ext_follower_n, f_ext_dead_n),
                max_steps=max_iter,
                kind="bounded" if max_iter is not None else "lax",
            )

        def load_step_func(
            i_step: int,
            init: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            hg_init, g_res_init = init[:2]

            # compute scaled external loads for this load step
            f_ext_follower_step = (
                f_ext_follower * (i_step + 1) / load_steps
                if f_ext_follower is not None
                else jnp.zeros((self.n_nodes, 6))
            )

            f_ext_dead_step = (
                f_ext_dead * (i_step + 1) / load_steps
                if f_ext_dead is not None
                else jnp.zeros((self.n_nodes, 6))
            )

            return inner_loop(f_ext_follower_step, f_ext_dead_step, g_res_init, hg_init)

        (
            hg,
            g_res,
        ) = jax.lax.fori_loop(
            0,
            load_steps,
            load_step_func,
            (
                self.hg0,
                jnp.zeros(n_solve_dofs),
                jnp.zeros((self.n_nodes, 6)),
                jnp.zeros((self.n_nodes, 6)),
            ),
        )[:2]

        d_n = self.make_d(hg)

        p_d = vmap(p, (0, 0, 0), 0)(
            d_n,
            self.ad_inv_o0[self.connectivity[:, 0], ...],
            self.ad_inv_o0[self.connectivity[:, 1], ...],
        )  # [n_elem, 6, 12]

        eps = self.make_eps(d_n)  # [n_elem, 6]

        f_int = self.make_f_int(p_d, eps).reshape(-1, 6)  # [n_dof]

        return StaticStructure(
            hg=hg,
            d=d_n,
            eps=eps,
            f_int=f_int,
            f_ext_follower=f_ext_follower,
            f_ext_dead=None,
        )
