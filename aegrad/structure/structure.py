from jax import numpy as jnp
from jax import Array, vmap
import jax
from jax.scipy.linalg import block_diag
from aegrad.utils import check_type
from aegrad.structure.data_structures import (
    StaticStructure,
    DynamicStructure,
    DynamicStructureSnapshot,
    ConvergenceStatus,
)
from aegrad.print_output import warn, warn_if_32_bit
from aegrad.structure.structure_utils import check_connectivity, n_elem_per_node
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.structure.structure_utils import (
    k_t_entry,
    integrate_m_l,
    integrate_c_t,
    make_c_t_lumped,
)
from aegrad.algebra.se3 import p, rmat_to_ha_hat, hg_to_d, exp_se3
from typing import Optional, Sequence, Literal
from functools import partial

from structure.time_integration import TimeIntregrator


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

        # allow for a single y_vector to be broadcast to all elements
        if y_vector.shape == (3,):
            y_vector = y_vector[None, :]
        if y_vector.shape == (1, 3):
            y_vector = jnp.broadcast_to(y_vector, (self.n_elem, 3))

        check_arr_shape(y_vector, (self.n_elem, 3), "y_vector")
        self.y_vector: Array = y_vector
        self.use_lumped_mass: bool = False

        # initialize design variables with default values
        self.x0: Array = jnp.zeros((num_nodes, 3))
        self.m: Array = jnp.zeros((self.n_elem, 6, 6))
        self.m_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_cs: Array = jnp.zeros((self.n_elem, 6, 6))
        self.k_l: Array = jnp.zeros((self.n_elem, 6, 6))
        self.m_lumped: Array = jnp.zeros((num_nodes, 6, 6))
        self.m_t_lumped: Array = jnp.zeros((self.n_dof, self.n_dof))

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
        self,
        coords: Array,
        k_cs: Array,
        m_cs: Optional[Array],
        m_lumped: Optional[Array] = None,
    ) -> None:
        r"""
        Set design variables and compute initial configuration dependent quantities
        :param coords: Node coordinates, [n_nodes, 3]
        :param k_cs: Cross-section stiffness matrices, [n_elem, 6, 6]
        :param m_cs: Cross-section mass matrices, [n_elem, 6, 6]
        :param m_lumped: Lumped mass matrices at nodes, [n_nodes, 6, 6]
        """
        # populate arrays
        self.k_cs = self.k_cs.at[...].set(k_cs)
        if m_cs is not None:
            self.m_cs = self.m_cs.at[...].set(m_cs)
        else:
            if self.use_gravity and m_lumped is None:
                warn(
                    "No mass matrices provided, but gravity is enabled. Assuming zero mass."
                )
        if m_lumped is not None:
            self.m_lumped = self.m_lumped.at[...].set(m_lumped)
            self.m_t_lumped = self.m_t_lumped.at[...].set(block_diag(*self.m_lumped))
            self.use_lumped_mass = True

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
            self.connectivity,
            self.o0,
            self.d0,
            jnp.zeros((self.n_elem, 6)),
            jnp.zeros((self.n_nodes, 6)),
            jnp.zeros((self.n_nodes, 6)),
            jnp.zeros((self.n_nodes, 6)),
            jnp.zeros((self.n_nodes, 6)),
        )

    def _assemble_matrix_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global matrix from element entries
        :param entries: Array of element matrix entries, [n_elem, 12, 12]
        :return: System global matrix, [n_dof, n_dof]
        """

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
        Assemble tangent mass matrix as a function of the element relative configuration vectors. This does not include
        the lumped mass contribution.
        :param d: Element relative configuration, [n_elem, 6]
        :param int_order: Integration order for mass matrix computation
        :return: Elementwise mass matrix, [n_elem, 12, 12]
        """
        return vmap(partial(integrate_m_l, int_order=int_order), (0, 0, 0, 0), 0)(
            self.m_cs, d, self.ad_inv_o0, self.l0
        )

    def _make_c_t(
        self, d: Array, d_dot: Array, v: Array, int_order: Literal[1, 2, 3] = 3
    ) -> tuple[Array, Array]:
        r"""
        Assemble tangent gyroscopic matrix. This does not include the lumped mass contribution.
        :param d: Element relative configuration, [n_elem, 6]
        :param d_dot: Element relative velocity, [n_elem, 6]
        :param v: Velocities in local frames, [n_node, 6]
        :param int_order: Integration order,
        :return: Elementwise gyroscopic C_L and C_T matrices, [n_elem, 12, 12], [n_elem, 12, 12]
        """
        clt = vmap(partial(integrate_c_t, int_order=int_order), (0, 0, 0, 0, 0, 0), 0)(
            self.m_cs,
            jnp.concatenate(
                (v[self.connectivity[:, 0], :], v[self.connectivity[:, 1], :]),
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

    def _make_c_t_lumped(self, v: Array) -> tuple[Array, Array]:
        r"""
        Obtain the gyroscopic matrix contribution from the lumped masses.
        :param v: Nodal velocities in global frame, [n_node, 6]
        :return: Gyroscopic L and T matrix entries from lumped masses, [n_node, 6, 6], [n_node, 6, 6]
        """
        c_t_l = vmap(make_c_t_lumped, (0, 0), 0)(self.m_lumped, v)  # [n_node, 2, 6, 6]
        return c_t_l[:, 0, :, :], c_t_l[:, 1, :, :]

    def _make_sys_matrix(
        self,
        m_t: Array,
        c_t: Array,
        c_t_lumped: Optional[Array],
        k_t: Array,
        ti: TimeIntregrator,
    ) -> Array:
        r"""
        Create the system matrix for the static or dynamic analysis.
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :param c_t: Disassembled system gyroscopic matrix, [n_elem, 12, 12]
        :param c_t_lumped: Disassembled system lumped gyroscopic matrix, [n_node, 6, 6]
        :param k_t: Disassembled system stiffness matrix postmultiplied by increment tangent, [n_elem, 12, 12]
        :param ti: Time integration parameters
        :return: System matrix, [n_dof, n_dof]
        """

        mat = self._assemble_matrix_from_entries(
            m_t * ti.beta_prime + c_t * ti.gamma_prime + k_t
        )

        if self.use_lumped_mass:
            mat += (
                self.m_t_lumped * ti.beta_prime
                + block_diag(*c_t_lumped) * ti.gamma_prime
            )

        return mat

    def _make_f_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Internal forces, [n_elem, 12]
        """

        return -jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs, eps)

    def _make_f_grav(self, m_t: Array, rmat: Array) -> Array:
        r"""
        Compute the global gravitational force vector. This does not include the lumped mass contribution.
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

    def _make_f_grav_lumped(self, rmat: Array) -> Array:
        r"""
        Compute the global gravitational force vector contribution from the lumped masses.
        :param rmat: Rotation matrices at nodes, [n_node, 3, 3]
        :return: Lumped gravity force vector, [n_node, 6]
        """
        f_rot = jnp.einsum("ikj,k->ij", rmat, self.gravity_vec)  # [n_node, 3]
        f_rot_tot = jnp.concatenate(
            (f_rot, jnp.zeros((self.n_nodes, 3))), axis=-1
        )  # [n_node, 6]
        return jnp.einsum("ijk,ik->ij", self.m_lumped, f_rot_tot)  # [n_node, 6]

    @staticmethod
    def _make_f_dead_ext(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, [n_node, 6]
        :param rmat: Deformation rotation matrices, [n_node, 3, 3]
        :return: External forces, [n_node, 6]
        """

        f_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        m_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        return jnp.concatenate((f_rot, m_rot), axis=-1)

    def _make_f_iner(self, m_l: Array, c_l: Array, v: Array, v_dot: Array) -> Array:
        r"""
        Compute the global inertial force vector.
        :param m_l: Disassembled system mass matrix, [n_elem, 12, 12]
        :param c_l: Disassembled system gyroscopic matrix, [n_elem, 12, 12]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :return: Inertial forces, [n_elem, 12]
        """

        v_elem = jnp.concatenate(
            (v[self.connectivity[:, 0], :], v[self.connectivity[:, 1], :]), axis=-1
        )  # [n_elem, 12]
        v_dot_elem = jnp.concatenate(
            (v_dot[self.connectivity[:, 0], :], v_dot[self.connectivity[:, 1], :]),
            axis=-1,
        )  # [n_elem, 12]

        return -jnp.einsum("ijk,ik->ij", m_l, v_dot_elem) - jnp.einsum(
            "ijk,ik->ij", c_l, v_elem
        )  # [n_elem, 12]

    def _make_f_iner_lumped(self, c_l_lumped: Array, v: Array, v_dot: Array) -> Array:
        r"""
        Obtain the contribution to the inertial forces from the lumped masses.
        :param c_l_lumped: Gyroscopic matrix from lumped masses, [n_node, 6, 6]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :return: Inertial forces from lumped masses, [n_node, 6]
        """
        return -jnp.einsum("ijk,ik->ij", self.m_lumped, v_dot) - jnp.einsum(
            "ijk,ik->ij", c_l_lumped, v
        )  # [n_node, 6]

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

    def _make_d_dot(self, p_d: Array, v: Array) -> Array:
        r"""
        Compute the time derivative of the element relative configuration vectors from the nodal velocities.
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :return: Element relative velocity vectors, [n_elem, 6]
        """

        v_elem = jnp.concatenate(
            (v[self.connectivity[:, 0], :], v[self.connectivity[:, 1], :]), axis=-1
        )  # [n_elem, 12]

        return jnp.einsum("ijk,ik->ij", p_d, v_elem)  # [n_elem, 6]

    def _resolve_forces(
        self,
        hg: Array,
        dynamic: bool,
        f_ext_dead: Optional[Array],
        v: Optional[Array],
        v_dot: Optional[Array],
    ) -> tuple[Array, Array, Optional[Array], Optional[Array], Array, Optional[Array]]:
        r"""
        Obtain all components of the force from a final solution
        :param hg: Nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        :param dynamic: Whether to compute dynamic forces
        :param f_ext_dead: External dead forces in global reference, [n_node, 6]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :return: Configuration vectors, strain vectors, Dead external forces, gravitational forces, internal forces and inertial forces
        """
        d = self._make_d(hg)
        eps = self._make_eps(d)
        p_d = self._make_p_d(d)
        d_dot = self._make_d_dot(p_d, v) if dynamic else None
        m_t = self._make_m_t(d) if (self.use_gravity or dynamic) else None
        c_l = self._make_c_t(d, d_dot, v)[0] if dynamic else None

        if self.use_lumped_mass and dynamic:
            c_l_lumped = self._make_c_t_lumped(v)[0]
        else:
            c_l_lumped = None

        if f_ext_dead is not None:
            this_f_ext_dead = self._make_f_dead_ext(f_ext_dead, hg[:, :3, :3])
        else:
            this_f_ext_dead = None

        if self.use_gravity:
            this_f_grav = self._assemble_vector_from_entries(
                self._make_f_grav(m_t, hg[:, :3, :3])
            ).reshape(-1, 6)
            if self.use_lumped_mass:
                this_f_grav += self._make_f_grav_lumped(hg[:, :3, :3])
        else:
            this_f_grav = None

        this_f_int = self._assemble_vector_from_entries(
            self._make_f_int(p_d, eps)
        ).reshape(-1, 6)

        if dynamic:
            this_f_iner = self._assemble_vector_from_entries(
                self._make_f_iner(m_t, c_l, v, v_dot)
            ).reshape(-1, 6)
            if self.use_lumped_mass:
                this_f_iner += self._make_f_iner_lumped(c_l_lumped, v, v_dot)
        else:
            this_f_iner = None

        return d, eps, this_f_ext_dead, this_f_grav, this_f_int, this_f_iner

    def _make_f_res(
        self,
        p_d: Array,
        eps: Array,
        hg: Array,
        f_ext_follower_n: Optional[Array],
        f_ext_dead_n: Optional[Array],
        dynamic: bool,
        m_t: Optional[Array],
        c_l: Optional[Array],
        c_l_lumped: Optional[Array],
        v: Optional[Array],
        v_dot: Optional[Array],
    ) -> Array:
        f_res = self._make_f_int(p_d, eps)  # [n_elem, 12]

        if self.use_gravity:
            f_res += self._make_f_grav(m_t, hg[:, :3, :3])

        if dynamic:
            f_res += self._make_f_iner(m_t, c_l, v, v_dot)

        f_res_vect = self._assemble_vector_from_entries(f_res)

        if f_ext_follower_n is not None:
            f_res_vect += f_ext_follower_n.reshape(self.n_dof).ravel()
        if f_ext_dead_n is not None:
            f_res_vect += self._make_f_dead_ext(f_ext_dead_n, hg[:, :3, :3]).ravel()

        if self.use_lumped_mass:
            if dynamic:
                f_res_vect += self._make_f_iner_lumped(c_l_lumped, v, v_dot).ravel()
            if self.use_gravity:
                f_res_vect += self._make_f_grav_lumped(hg[:, :3, :3]).ravel()

        return f_res_vect  # [n_dof]

    @staticmethod
    def _update_hg(hg: Array, d_ha: Array) -> Array:
        r"""
        Update the nodal homogeneous transformation matrices with the configuration increments.
        :param hg: Existing nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        :param d_ha: Perturbation to the configuration vector, [n_nodes, 6]
        :return: Updated nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        """
        return jnp.einsum(
            "ijk,ikl->ijl",
            hg,
            vmap(exp_se3, 0, 0)(d_ha.reshape(-1, 6)),
        )

    def get_cg(self, hg: Array):
        r"""
        Compute the total mass and the coordinate of center of gravity.
        :param hg: Node locations in SE(3), [n_nodes, 4, 4]
        :return: Total mass, x coordinate of center of gravity, [3]
        """

        # TODO: implement
        raise NotImplementedError

    def make_prescribed_dofs_array(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
    ) -> Array:
        # degrees of freedom which are prescribed
        if isinstance(prescribed_dofs, slice) or isinstance(prescribed_dofs, Sequence):
            return jnp.arange(self.n_dof)[prescribed_dofs]
        elif isinstance(prescribed_dofs, int):
            return jnp.array([prescribed_dofs])
        elif isinstance(prescribed_dofs, Array):
            return prescribed_dofs
        elif prescribed_dofs is None:
            return jnp.array([], dtype=int)
        else:
            raise TypeError(
                "prescribed_dofs must be an int, slice, Sequence[int], or Array"
            )

    def static_solve(
        self,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        prescribed_dofs: Sequence[int] | Array | slice | int,
        include_material: bool = True,
        include_geometric: bool = False,
        load_steps: int = 1,
        relaxation_factor: float = 1.0,
        max_iter: Optional[int] = 40,
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
        :param relaxation_factor: Relaxation factor for updates, in range (0, 1].
        :param max_iter: Maximum number of Newton-Raphson iterations per load step.
        :param abs_tol: Absolute tolerance for convergence.
        :return: StaticStructure dataclass containing results of the static analysis.
        """

        if load_steps < 1:
            raise ValueError("load_steps must be at least 1")

        # add a warning if using 32-bit floats
        warn_if_32_bit()

        # check inputs
        if f_ext_follower is not None:
            check_arr_shape(f_ext_follower, (self.n_nodes, 6), "f_ext_follower")
        if f_ext_dead is not None:
            check_arr_shape(f_ext_dead, (self.n_nodes, 6), "f_ext_dead")

        if not (0.0 < relaxation_factor <= 1.0):
            raise ValueError("relaxation_factor must be in the range (0, 1]")

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)
        n_solve_dofs = solve_dofs.shape[0]

        # process external forces for load stepping
        load_step_weight: Array = jnp.linspace(0.0, 1.0, load_steps + 1)[
            1:
        ]  # [load_steps]
        if f_ext_follower is not None:
            f_ext_follower_steps = jnp.einsum(
                "i,jk->ijk", load_step_weight, f_ext_follower
            )  # [load_steps, n_node, 6]
        else:
            f_ext_follower_steps = None

        if f_ext_dead is not None:
            f_ext_dead_steps = jnp.einsum(
                "i,jk->ijk", load_step_weight, f_ext_dead
            )  # [load_steps, n_node, 6]
        else:
            f_ext_dead_steps = None

        def _update(
            i_load_step: int,
            converge_status_n: ConvergenceStatus,
            hg_n: Array,
        ) -> tuple[int, ConvergenceStatus, Array]:
            # base parameters
            d_n = self._make_d(hg_n)  # [n_elem, 6]
            p_d_n = self._make_p_d(d_n)  # [n_elem, 6, 12]
            eps_n = self._make_eps(d_n)  # [n_elem, 6]

            # assemble tangent stiffness matrix, [n_solve_dof, n_solve_dof]
            k_t_n = self._assemble_matrix_from_entries(
                self._make_k_t(d_n, p_d_n, eps_n, include_material, include_geometric)
            )[jnp.ix_(solve_dofs, solve_dofs)]

            # compute residual forces, [n_solve_dofs]
            f_res_n_solve = self._make_f_res(
                p_d_n,
                eps_n,
                hg_n,
                f_ext_follower_steps[i_load_step, ...]
                if f_ext_follower is not None
                else None,
                f_ext_dead_steps[i_load_step, ...] if f_ext_dead is not None else None,
                False,
                self._make_m_t(d_n) if self.use_gravity else None,
                None,
                None,
                None,
                None,
            )[solve_dofs]

            # solve for configuration increment, [n_solve_dofs]
            d_ha_np1 = jnp.linalg.solve(k_t_n, f_res_n_solve) * relaxation_factor
            d_ha_np1_full = jnp.zeros(self.n_dof)
            d_ha_np1_full = d_ha_np1_full.at[solve_dofs].set(d_ha_np1)

            # update configuration, [n_nodes, 4, 4]
            hg_np1_full = self._update_hg(hg_n, d_ha_np1_full)

            # update convergence status
            converge_status_np1 = ConvergenceStatus(
                d_ha_np1,
                converge_status_n.abs_tol,
                converge_status_n.i_iter + 1,
                converge_status_n.n_iter,
            )
            return i_load_step, converge_status_np1, hg_np1_full

        def inner_loop(
            i_load_step: int,
            hg_init: Array,
        ) -> Array:
            _, convergence_status, hg_solve = jax.lax.while_loop(
                lambda args_: ~args_[1].get_status(),
                lambda args_: _update(*args_),
                (
                    i_load_step,
                    ConvergenceStatus(
                        jnp.zeros(n_solve_dofs),
                        jnp.array(abs_tol),
                        jnp.zeros((), dtype=int),
                        jnp.array(max_iter, dtype=int)
                        if max_iter is not None
                        else None,
                    ),
                    hg_init,
                ),
            )

            convergence_status.print_message(None, i_load_step)

            return hg_solve

        def load_step_func(
            i_step: int,
            hg_: Array,
        ) -> Array:
            return inner_loop(i_step, hg_)

        hg = jax.lax.fori_loop(
            0,
            load_steps,
            load_step_func,
            self.hg0,
        )

        # postprocess final results
        d, eps, f_ext_dead_local, f_grav, f_int, _ = self._resolve_forces(
            hg, False, f_ext_dead, None, None
        )

        return StaticStructure(
            hg=hg,
            conn=self.connectivity,
            o0=self.o0,
            d=d,
            eps=eps,
            f_int=f_int,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead_local,
            f_grav=f_grav,
        )

    def dynamic_solve(
        self,
        init_state: Optional[DynamicStructureSnapshot | StaticStructure],
        n_tstep: int,
        dt: Array | float,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        prescribed_dofs: Sequence[int] | Array | slice | int | None,
        include_material: bool = True,
        include_geometric: bool = False,
        relaxation_factor: float = 1.0,
        max_iter: int = 40,
        spectral_radius: float = 1.0,
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
        :param relaxation_factor: Relaxation factor for Newton-Raphson iterations, in the range (0, 1].
        :param max_iter: Maximum number of Newton-Raphson iterations per load step.
        :param abs_tol: Absolute tolerance for convergence.
        :return: DynamicStructure dataclass containing results of the dynamic analysis.
        """

        # add a warning if using 32-bit floats
        warn_if_32_bit()

        # set up initial state
        if init_state is None:
            init_state_: DynamicStructureSnapshot = (
                self.reference_configuration().to_dynamic()
            )
        elif isinstance(init_state, StaticStructure):
            init_state_ = init_state.to_dynamic()
        else:
            init_state_ = init_state

        if not (0.0 < relaxation_factor <= 1.0):
            raise ValueError("relaxation_factor must be in the range (0, 1]")

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)
        n_solve_dofs = len(solve_dofs)

        # check and process external forces
        def check_force(arr: Optional[Array], name: str) -> Optional[Array]:
            if arr is None:
                return None
            match arr.ndim:
                case 2:
                    out = jnp.broadcast_to(arr[None, ...], (n_tstep, self.n_nodes, 6))
                case 3:
                    out = arr
                case _:
                    raise ValueError(
                        f"{name} must have shape [n_node, 6] or [n_tstep, n_node, 6]"
                    )
            check_arr_shape(out, (n_tstep, self.n_nodes, 6), name)
            return out

        f_ext_dead = check_force(f_ext_dead, "f_ext_dead")  # [n_tstep, n_node, 6]
        f_ext_follower = check_force(
            f_ext_follower, "f_ext_follower"
        )  # [n_tstep, n_node, 6]

        # time integration parameters
        dt: Array = jnp.array(dt)
        time_integrator = TimeIntregrator(spectral_radius=spectral_radius, dt=dt)

        def _update(
            i_ts: int,
            converge_status_n: ConvergenceStatus,
            hg_n: Array,
            n_n: Array,
            v_n: Array,
            v_dot_n: Array,
        ) -> tuple[
            int,
            ConvergenceStatus,
            Array,
            Array,
            Array,
            Array,
        ]:
            # TODO: rewrite to make more efficient by avoiding recomputation of HG

            hg_update = self._update_hg(hg_n, n_n)  # [n_node, 4, 4]

            # base parameters
            d_n = self._make_d(hg_update)  # [n_elem, 6]
            p_d_n = self._make_p_d(d_n)  # [n_elem, 6, 12]
            eps_n = self._make_eps(d_n)  # [n_elem, 6]
            d_dot_n = self._make_d_dot(p_d_n, v_n)  # [n_elem, 6]

            # tangent matrices
            m_t = self._make_m_t(d_n)  # [n_elem, 12, 12]
            c_l, c_t = self._make_c_t(
                d_n, d_dot_n, v_n
            )  # [n_elem, 12, 12], [n_elem, 12, 12]
            k_t = self._make_k_t(
                d_n, p_d_n, eps_n, include_material, include_geometric
            )  # [n_elem, 12, 12]

            if self.use_lumped_mass:
                c_l_lumped, c_t_lumped = self._make_c_t_lumped(
                    v_n
                )  # [n_node, 6, 6], [n_node, 6, 6]
            else:
                c_l_lumped, c_t_lumped = None, None

            # residual forces, [n_solve_dofs]
            f_res_n_solve = self._make_f_res(
                p_d_n,
                eps_n,
                hg_n,
                f_ext_follower[i_ts, ...] if f_ext_follower is not None else None,
                f_ext_dead[i_ts, ...] if f_ext_dead is not None else None,
                True,
                m_t,
                c_l,
                c_l_lumped,
                v_n,
                v_dot_n,
            )[solve_dofs]

            # system matrix, [n_solve_dofs, n_solve_dofs]
            sys_mat = self._make_sys_matrix(m_t, c_t, c_t_lumped, k_t, time_integrator)[
                jnp.ix_(solve_dofs, solve_dofs)
            ]

            # solve for configuration increment, [n_solve_dofs]
            d_n_np1 = jnp.linalg.solve(sys_mat, f_res_n_solve) * relaxation_factor
            n_np1 = n_n.ravel().at[solve_dofs].add(d_n_np1).reshape(-1, 6)

            # update configuration, velocities and accelerations
            v_np1 = (
                v_n.ravel()
                .at[solve_dofs]
                .add(time_integrator.gamma_prime * d_n_np1)
                .reshape(-1, 6)
            )
            v_dot_np1 = (
                v_dot_n.ravel()
                .at[solve_dofs]
                .add(time_integrator.beta_prime * d_n_np1)
                .reshape(-1, 6)
            )

            # update convergence status
            converge_status_np1 = ConvergenceStatus(
                d_n_np1,
                converge_status_n.abs_tol,
                converge_status_n.i_iter + 1,
                converge_status_n.n_iter,
            )

            return (
                i_ts,
                converge_status_np1,
                hg_n,
                n_np1,
                v_np1,
                v_dot_np1,
            )

        def inner_loop(
            i_ts: int, sol: DynamicStructureSnapshot
        ) -> DynamicStructureSnapshot:
            a_init = time_integrator.predict_a(
                sol.v_dot[i_ts - 1, ...], sol.a[i_ts - 1, ...]
            )

            n_init = time_integrator.predict_n(
                sol.v[i_ts - 1, ...], sol.a[i_ts - 1, ...], a_init
            )
            v_init = time_integrator.predict_v(
                sol.v[i_ts - 1, ...], sol.a[i_ts - 1, ...], a_init
            )
            v_dot_init = time_integrator.predict_v_dot(
                sol.v_dot[i_ts - 1, ...], sol.a[i_ts - 1], a_init
            )

            _, converge_status, hg, n, v, v_dot = jax.lax.while_loop(
                lambda args_: ~args_[1].get_status(),
                lambda args_: _update(*args_),
                (
                    i_ts,
                    ConvergenceStatus(
                        jnp.zeros(n_solve_dofs),
                        jnp.array(abs_tol),
                        jnp.zeros((), dtype=int),
                        jnp.array(max_iter, dtype=int)
                        if max_iter is not None
                        else None,
                    ),
                    sol.hg[i_ts - 1, ...],
                    n_init,
                    v_init,
                    v_dot_init,
                ),
            )

            converge_status.print_message(i_ts, None)

            hg_np1 = self._update_hg(sol.hg[i_ts - 1, ...], n)
            d, eps, f_ext_dead_local, f_grav, f_int, f_iner = self._resolve_forces(
                hg_np1,
                True,
                f_ext_dead[i_ts, ...] if f_ext_dead is not None else None,
                v,
                v_dot,
            )
            sol.d = sol.d.at[i_ts, ...].set(d)
            sol.eps = sol.eps.at[i_ts, ...].set(eps)
            sol.v = sol.v.at[i_ts, ...].set(v)
            sol.v_dot = sol.v_dot.at[i_ts, ...].set(v_dot)
            sol.hg = sol.hg.at[i_ts, ...].set(hg_np1)
            if f_ext_follower is not None:
                sol.f_ext_follower = sol.f_ext_follower.at[i_ts, ...].set(
                    f_ext_follower[i_ts, ...]
                )
            if f_ext_dead is not None:
                sol.f_ext_dead = sol.f_ext_dead.at[i_ts, ...].set(f_ext_dead_local)
            if self.use_gravity:
                sol.f_grav = sol.f_grav.at[i_ts, ...].set(f_grav)
            sol.f_int = sol.f_int.at[i_ts, ...].set(f_int)
            sol.f_iner = sol.f_iner.at[i_ts, ...].set(f_iner)

            # update pseudoacceleration
            sol.a = sol.a.at[i_ts, ...].set(
                time_integrator.calculate_a_np1(
                    sol.v_dot[i_ts, ...], sol.v_dot[i_ts - 1, ...], sol.a[i_ts - 1, ...]
                )
            )

            return sol

        # test initial state to ensure that equilibrium is satisfied
        def evaluate_initial_equilibrium(
            init_state_: DynamicStructureSnapshot,
        ) -> DynamicStructureSnapshot:
            d, eps, f_ext_dead_, f_grav, f_int, f_iner = self._resolve_forces(
                init_state_.hg,
                True,
                init_state_.f_ext_dead,
                init_state_.v,
                init_state_.v_dot,
            )

            f_res = f_int + f_iner
            if f_grav is not None:
                f_res += f_grav
            if f_ext_dead_ is not None:
                f_res += f_ext_dead_
            if init_state_.f_ext_follower is not None:
                f_res += init_state_.f_ext_follower

            max_res = jnp.max(jnp.abs(f_res))
            if max_res > 1e-6:
                warn(
                    f"Initial state is not in equilibrium, maximum residual force is {max_res:.3e}"
                )

            return DynamicStructureSnapshot(
                hg=init_state_.hg,
                conn=self.connectivity,
                o0=self.o0,
                d=d,
                eps=eps,
                v=init_state_.v,
                v_dot=init_state_.v_dot,
                a=init_state_.v_dot,  # initial pseudoacceleration set equal to initial acceleration
                f_ext_follower=init_state_.f_ext_follower,
                f_ext_dead=f_ext_dead_,
                f_grav=f_grav,
                f_int=f_int,
                f_iner=f_iner,
                t=init_state_.t,
                i_ts=init_state_.i_ts,
            )

        # obtain values for timestep 0
        init_state_eval = evaluate_initial_equilibrium(init_state_)

        # solve
        t = jnp.arange(n_tstep) * dt + init_state_.t
        output = jax.lax.fori_loop(
            1, n_tstep, inner_loop, DynamicStructure.initialise(init_state_eval, t)
        )

        return output
