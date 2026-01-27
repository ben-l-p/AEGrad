from jax import numpy as jnp
from jax import Array, vmap
import jax
import equinox
from aegrad.utils import check_type
from aegrad.structure.data_structures import (
    StaticStructure,
    DynamicStructure,
    DynamicStructureSnapshot,
)
from aegrad.structure.structure_utils import check_connectivity, n_elem_per_node
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.structure.structure_utils import k_t_entry, integrate_m_l, integrate_c_t
from aegrad.algebra.se3 import p, rmat_to_ha_hat, hg_to_d, exp_se3
from typing import Optional, Sequence, Literal
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
        gravity: Optional[Array] = None,
    ) -> None:
        r"""
        Initialise Structure class with all non-design parameters
        :param num_nodes: Number of nodes in the structure
        :param connectivity: Connectivity array of shapes [n_elem, 2]
        :param y_vector: Vector defining the y direction for each element, [n_elem, 3]
        :param gravity: Gravity vector in global reference frame, or None for no gravity_vec, [3]
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

        # gravity_vec settings
        self.use_gravity: bool = gravity is not None and jnp.any(gravity)
        if self.use_gravity:
            check_arr_shape(gravity, (3,), "gravity")
            self.gravity_vec: Array = gravity
        else:
            self.gravity_vec = jnp.zeros((3,))

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

    def reference_configuration(self) -> StaticStructure:
        r"""
        Get the reference configuration of the structure
        :return: StaticStructure dataclass containing reference configuration
        """
        return StaticStructure(
            self.hg0,
            self.d0,
            jnp.zeros((self.n_elem, 6)),
            jnp.zeros((self.n_nodes, 6)),
            jnp.zeros((self.n_nodes, 6)),
            jnp.zeros((self.n_nodes, 6)),
        )

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

    def _assemble_vector_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global vector from element entries
        :param entries: Array of element vector entries, [n_elem, 12]
        :return: System global vector, [n_dof]
        """

        vect = jnp.zeros(self.n_dof)
        vect = vect.at[self.dof_per_elem[:, :6]].add(entries[:, :6])
        return vect.at[self.dof_per_elem[:, 6:]].add(entries[:, 6:])

    def _make_k_t(
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
        :return: Elementwise stiffness matrix entries, [n_elem, 12, 12]
        """
        # compute stiffness matrix entries
        return vmap(
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

    def _make_m_t(self, d: Array, int_order: Literal[3, 4, 5] = 3) -> Array:
        r"""
        Assemble tangent mass matrix as a function of the element relative configuration vectors
        :param d: Element relative configuration, [n_elem, 6]
        :param int_order: Integration order for mass matrix computation
        :return: Elementwise mass matrix, [n_elem, 12, 12]
        """
        return vmap(partial(integrate_m_l, int_order=int_order), (0, 0, 0, 0), 0)(
            self.m_cs, d, self.ad_inv_o0, self.l0
        )

    def _make_c_t(
        self, d: Array, d_dot: Array, v_ab: Array, int_order: Literal[1, 2, 3] = 3
    ) -> tuple[Array, Array]:
        r"""
        Assemble tangent gyroscopic matrix.
        :param d: Element relative configuration, [n_elem, 6]
        :param d_dot: Element relative velocity, [n_elem, 6]
        :param v_ab: Velocities in local frames, [n_node, 6]
        :param int_order: Integration order,
        :return: Elementwise gyroscopic C_L and C_T matrices, [n_elem, 12, 12], [n_elem, 12, 12]
        """
        clt = vmap(partial(integrate_c_t, int_order=int_order), (0, 0, 0, 0, 0, 0), 0)(
            self.m_cs,
            jnp.concatenate(
                (v_ab[self.connectivity[:, 0], :], v_ab[self.connectivity[:, 1], :]),
                axis=-1,
            ),
            d,
            d_dot,
            self.ad_inv_o0,
            self.l0,
        )
        cl = clt[:, 0, ...]
        ct = clt[:, 1, ...]
        return cl, ct

    # def _make_sys_matrix(
    #     self,
    #     d: Array,
    #     v_ab: Array,
    #     p_d: Array,
    #     eps: Array,
    #     gamma_prime: float,
    #     beta_prime: float,
    # ) -> Array:
    #     r"""
    #     Create the system matrix for the static or dynamic analysis.
    #     :param d: Element relative configuration, [n_elem, 6]
    #     :param v_ab: Velocities in local frames, [n_node, 6]
    #     :param p_d: P(d) operator, [n_elem, 6, 12]
    #     :param eps: Element strain vectors, [n_elem, 6]
    #     :return: System matrix, [n_dof, n_dof]
    #     """
    #
    #     d_dot = jnp.einsum(
    #         "ijk,ik->ij",
    #         p_d,
    #         jnp.concatenate(
    #             (v_ab[self.connectivity[:, 0], :], v_ab[self.connectivity[:, 1], :])
    #         ),
    #     )  # [n_elem, 6]
    #
    #     m_t = self._make_m_t(d)
    #     c_t = self._make_c_t(d, d_dot, v_ab)
    #     k_t = self._make_k_t(d, p_d, eps, True, True)
    #
    #     return m_t * beta_prime + c_t * gamma_prime + k_t

    def _make_f_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Internal forces, [n_elem, 12]
        """

        return jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs, eps)

    def _make_f_grav(self, m_t: Array, rmat: Array) -> Array:
        r"""
        Compute the global gravitational force vector.
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :param rmat: Node rotations from reference, [n_node, 3, 3]
        :return: Gravity force element vector, [n_elem, 12]
        """
        f_rot = jnp.einsum("ikj,k->ij", rmat, self.gravity_vec)  # [n_node, 3]
        f_rot_tot = jnp.concatenate(
            (f_rot, jnp.zeros((self.n_nodes, 3))), axis=-1
        )  # [n_node, 6]
        return jnp.einsum(
            "ijk,ik->ij",
            m_t,
            jnp.concatenate(
                (
                    f_rot_tot[self.connectivity[:, 0], :],
                    f_rot_tot[self.connectivity[:, 1], :],
                ),
                axis=1,
            ),
        )  # [n_elem, 12]

    @staticmethod
    def _make_f_dead_ext(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, [n_node, 6]
        :param rmat: Deformation rotation matrices, [n_node, 3, 3]
        :return: External forces, [n_elem, 6]
        """

        # f_rot = jnp.einsum("ijk,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        # m_rot = jnp.einsum("ijk,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        f_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        m_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        return jnp.concatenate((f_rot, m_rot), axis=-1)

    def _make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Element strain vectors, [n_elem, 6]
        """

        return (d - self.d0) / self.l0[:, None]

    def _make_p_d(self, d: Array) -> Array:
        r"""
        Compute the P(d) operator as a function of the element relative configuration vectors.
        :param d: Relative configuration vectors, [n_elem, 6]
        :return: P(d) operator, [n_elem, 6, 12]
        """
        return vmap(p, (0, 0), 0)(d, self.ad_inv_o0)  # [n_elem, 6, 12]

    def _make_d(self, hg: Array) -> Array:
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

    def make_f_res(
        self,
        p_d: Array,
        eps: Array,
        hg: Array,
        f_ext_follower_n: Optional[Array],
        f_ext_dead_n: Optional[Array],
        m_t: Optional[Array],
    ) -> Array:
        f_res = self._make_f_int(p_d, eps)  # [n_elem, 12]

        if self.use_gravity:
            f_res -= self._assemble_vector_from_entries(
                self._make_f_grav(m_t, hg[:, :3, :3])
            )

        f_res_vect = self._assemble_vector_from_entries(f_res)

        if f_ext_follower_n is not None:
            f_res_vect -= f_ext_follower_n.reshape(self.n_dof).ravel()
        if f_ext_dead_n is not None:
            f_res_vect -= self._make_f_dead_ext(f_ext_dead_n, hg[:, :3, :3]).ravel()

        return f_res_vect  # [n_dof]

    @staticmethod
    def _update_hg(hg: Array, d_ha: Array) -> Array:
        return jnp.einsum(
            "ijk,ikl->ijl",
            hg,
            vmap(exp_se3, 0, 0)(d_ha.reshape(-1, 6)),
        )

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

        def _update(
            hg_n_full: Array,
            f_res_n: Array,
            f_ext_follower_n: Array,
            f_ext_dead_n: Array,
        ) -> tuple[Array, Array, Array, Array]:
            # configuration vectors, [n_elem, 6]
            d_n = self._make_d(hg_n_full)
            p_d_n = self._make_p_d(d_n)
            eps_n = self._make_eps(d_n)  # [n_elem, 6]

            k_t_n = self.assemble_matrix_from_entries(
                self._make_k_t(d_n, p_d_n, eps_n, include_material, include_geometric)
            )

            m_t = self._make_m_t(d_n) if self.use_gravity else None
            f_res_n_solve = self.make_f_res(
                p_d_n, eps_n, hg_n_full, f_ext_follower_n, f_ext_dead_n, m_t
            )[solve_dofs]

            d_ha_np1 = -jnp.linalg.solve(
                k_t_n[jnp.ix_(solve_dofs, solve_dofs)], f_res_n_solve
            )
            d_ha_np1_full = jnp.zeros(self.n_dof)
            d_ha_np1_full = d_ha_np1_full.at[solve_dofs].set(d_ha_np1)

            hg_np1_full = self._update_hg(hg_n_full, d_ha_np1_full)
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
                lambda args: _update(*args),
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

        d_n_ = self._make_d(hg)
        p_d_ = self._make_p_d(d_n_)
        eps_ = self._make_eps(d_n_)  # [n_elem, 6]

        f_int_ = self._assemble_vector_from_entries(
            self._make_f_int(p_d_, eps_)
        ).reshape(-1, 6)  # [n_node, 6]

        return StaticStructure(
            hg=hg,
            d=d_n_,
            eps=eps_,
            f_int=f_int_,
            f_ext_follower=f_ext_follower,
            f_ext_dead=None,
        )

    def dynamic_solve(
        self,
        init_state: Optional[DynamicStructureSnapshot | StaticStructure],
        n_tstep: int,
        dt: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        prescribed_dofs: Sequence[int] | Array | slice | int,
        include_material: bool = True,
        include_geometric: bool = False,
        load_steps: int = 1,
        max_iter: int = 40,
        spectral_radius: float = 0.9,
        rel_tol: float = 1e-7,
        abs_tol: float = 1e-9,
    ) -> DynamicStructure:
        r"""
        Perform dynamic solve of the structure under external loads
        :param init_state: Initial state of the structure, either as a DynamicStructureSnapshot or StaticStructure. If
        None, the reference configuration is used with zero velocities.
        :param n_tstep: Number of time steps to simulate
        :param dt: Time step length
        :param f_ext_follower: Following external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external follower forces
        :param f_ext_dead: Dead external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external dead forces
        :param prescribed_dofs: Degrees of freedom which are prescribed (not solved for).
        :param include_material: If True, include material stiffness contribution.
        :param include_geometric: If True, include geometric stiffness contribution.
        :param load_steps: Number of load steps to apply the external loads over.
        :param max_iter: Maximum number of Newton-Raphson iterations per load step.
        :param rel_tol: Relative tolerance for convergence. measured for resultant forces relative to maximum applied
        force.
        :param abs_tol: Absolute tolerance for convergence.
        :return: DynamicStructure dataclass containing results of the dynamic analysis.
        """

    #     # set up initial state
    #     if init_state is None:
    #         init_state_: DynamicStructureSnapshot = (
    #             self.reference_configuration().to_dynamic()
    #         )
    #     elif isinstance(init_state, StaticStructure):
    #         init_state_ = init_state.to_dynamic()
    #     else:
    #         init_state_ = init_state
    #
    #     # check and process external forces
    #     def check_force(arr: Optional[Array], name: str) -> Optional[Array]:
    #         if arr is None:
    #             return None
    #         match arr.ndim:
    #             case 2:
    #                 out = jnp.broadcast_to(arr[None, ...], (n_tstep, self.n_nodes, 6))
    #             case 3:
    #                 out = arr
    #             case _:
    #                 raise ValueError(
    #                     f"{name} must have shape [n_node, 6] or [n_tstep, n_node, 6]"
    #                 )
    #         check_arr_shape(out, (n_tstep, self.n_nodes, 6), name)
    #         return out
    #
    # f_ext_dead = check_force(f_ext_dead, "f_ext_dead")  # [n_tstep, n_node, 6]
    # f_ext_follower = check_force(
    #     f_ext_follower, "f_ext_follower"
    # )  # [n_tstep, n_node, 6]
    #
    # time integration parameters
    # gamma_prime, beta_prime = get_integration_parameters(spectral_radius, dt)
    #
    # result = DynamicStructure.initialise(init_state_, n_tstep)
    #
    # def inner_loop(
    #     hg: Array, d: Array, p_d: Array, eps: Array, v: Array, v_dot: Array
    # ) -> Array:
    #     a_sys = self._make_sys_matrix(
    #         d,
    #         v,
    #         p_d,
    #         eps,
    #         gamma_prime,
    #         beta_prime,
    #     )
    #
    #     f_res =
    #
    # def timestep_loop(i_ts: int, res_obj: DynamicStructure) -> DynamicStructure:
    #     pass
