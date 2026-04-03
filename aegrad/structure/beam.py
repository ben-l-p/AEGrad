import copy
from typing import Optional, Sequence, Literal, overload
from functools import partial

from jax import numpy as jnp
from jax import Array, vmap
import jax
from jax.scipy.linalg import block_diag

from utils import _check_type
from structure.data_structures import (
    StaticStructure,
    DynamicStructure,
    DynamicStructureSnapshot,
    OptionalJacobians,
)
from data_structures import ConvergenceSettings, ConvergenceStatus
from print_utils import warn, warn_if_32_bit, VerbosityLevel
from structure.utils import _check_connectivity, _n_elem_per_node
from algebra.array_utils import check_arr_shape, check_arr_dtype
from structure.utils import (
    _k_t_entry,
    _integrate_m_l,
    _integrate_c_t,
    _make_c_t_lumped,
)
from algebra.se3 import p, rmat_to_ha_hat, hg_to_d, exp_se3
from algebra.so3 import vec_to_skew
from structure.time_integration import TimeIntregrator
from algebra.se3 import t_se3, log_se3
from structure.gradients.data_structures import StructuralDesignVariables
from structure.data_structures import StructureMinimalStates


class BaseBeamStructure:
    r"""
    Class to represent nonlinear beam structure_dv model
    """

    def __init__(
            self,
            num_nodes: int,
            connectivity: Array,
            y_vector: Array,
            gravity: Optional[Array] = None,
            verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
            optional_jacobians: Optional[OptionalJacobians] = None,
            convergence_settings: Optional[ConvergenceSettings] = None,
    ) -> None:
        r"""
        Initialise BaseBeamStructure class with all non-design parameters
        :param num_nodes: Number of nodes in the structure_dv
        :param connectivity: Connectivity array of arr_list_shapes [n_elem, 2]
        :param y_vector: Vector defining the y direction for each element, [n_elem, 3]
        :param gravity: Gravity vector in global reference frame, or None for no gravity_vec, [3]
        """

        _check_type(num_nodes, int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        check_arr_shape(connectivity, (None, 2), "connectivity")
        check_arr_dtype(connectivity, int, "connectivity")
        _check_connectivity(connectivity, num_nodes)
        self.connectivity: Array = connectivity  # [n_elem, 2]
        self.n_elem_per_node: Array = _n_elem_per_node(connectivity)  # [n_nodes_]
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
        self.m_lumped: Array = jnp.zeros((num_nodes, 6, 6))
        self.m_t_lumped: Array = jnp.zeros((self.n_dof, self.n_dof))

        # initialise auxiliary arrays
        self.o0: Array = jnp.zeros((self.n_elem, 3, 3))
        self.l0: Array = jnp.zeros(self.n_elem)
        self.d0: Array = jnp.zeros((self.n_elem, 6))

        # initialise undeformed algebra and group
        self.hg0: Array = jnp.zeros((self.n_nodes, 4, 4))

        # grads inverse action for the reference rotations
        self.ad_inv_o0: Array = jnp.zeros((self.n_elem, 6, 6))

        # gravity_vec settings
        self.use_gravity: bool = gravity is not None and bool(jnp.any(gravity))
        if self.use_gravity:
            check_arr_shape(gravity, (3,), "gravity")  # type: ignore
            self.gravity_vec: Array = gravity  # type: ignore
        else:
            self.gravity_vec = jnp.zeros((3,))

        # other settings
        self.verbosity: VerbosityLevel = verbosity
        self.use_m_cs: bool = False

        self.optional_jacobians: OptionalJacobians = (
            optional_jacobians
            if optional_jacobians is not None
            else OptionalJacobians()
        )
        self.convergence_settings: ConvergenceSettings = (
            convergence_settings
            if convergence_settings is not None
            else ConvergenceSettings()
        )

        self._time_integrator: Optional[TimeIntregrator] = None

    @property
    def time_integrator(self) -> TimeIntregrator:
        if self._time_integrator is None:
            raise ValueError(
                "Time integrator has not been set. Please set time_integrator before accessing."
            )
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, ti: TimeIntregrator) -> None:
        self._time_integrator = ti

    def set_design_variables(
            self,
            coords: Array,
            k_cs: Array,
            m_cs: Optional[Array],
            m_lumped: Optional[Array] = None,
            *,
            remove_checks: bool = False,
    ) -> None:
        r"""
        Set design variables and compute initial configuration dependent quantities
        :param coords: Node coordinates, [n_nodes_, 3]
        :param k_cs: Cross-section stiffness matrices, [n_elem, 6, 6]
        :param m_cs: Cross-section mass matrices, [n_elem, 6, 6]
        :param m_lumped: Lumped mass matrices at nodes, [n_nodes_, 6, 6]
        :param remove_checks: Flag to ignore input checks, used when Jitted.
        """
        # populate arrays
        self.k_cs = self.k_cs.at[...].set(k_cs)
        if m_cs is not None:
            self.m_cs = self.m_cs.at[...].set(m_cs)
            self.use_m_cs: bool = True
        else:
            self.use_m_cs = False
            if self.use_gravity and m_lumped is None:
                warn(
                    "No mass matrices provided, but gravity is enabled. Assuming zero mass.",
                    current_level=self.verbosity,
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
        if not remove_checks:
            if jnp.any(
                    jnp.linalg.norm(jnp.cross(dx, self.y_vector, 1, 1), axis=-1) < 1e-6
            ):
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
        )  # [n_nodes_, 4, 4]
        self.hg0 = self.hg0.at[:, :3, 3].set(self.x0)

    def get_design_variables(self, struct_case: StaticStructure | DynamicStructure) -> StructuralDesignVariables:
        r"""
        Obtain the design variables for the structural problem. As the external forcing is defined for each solve, the
        chosen forcing is required as input.
        :param f_ext_follower: External forcing follower, [n_tstep, n_nodes, 6] or [n_nodes, 6]
        :param f_ext_dead: External forcing dead, [n_tstep, n_nodes, 6] or [n_nodes, 6]
        :return: StructuralDesignVariables dataclass containing design variables
        """

        gs = copy.deepcopy(struct_case)
        gs.to_global()
        return StructuralDesignVariables(x0=self.x0, m_cs=self.m_cs, k_cs=self.k_cs, m_lumped=self.m_lumped,
                                         f_ext_dead=gs.f_ext_dead, f_ext_follower=gs.f_ext_follower)

    def reference_configuration(
            self,
            use_f_ext_follower: bool = True,
            use_f_ext_dead: bool = True,
            use_f_aero: bool = True,
            use_f_grav: bool = True,
            prescribed_dofs: Optional[Array] = None,
    ) -> StaticStructure:
        r"""
        Get the reference configuration of the structure_dv
        :return: StaticStructure dataclass containing reference configuration
        """
        return StaticStructure(
            hg=self.hg0,
            conn=self.connectivity,
            o0=self.o0,
            d=self.d0,
            eps=jnp.zeros((self.n_elem, 6)),
            varphi=jnp.zeros((self.n_nodes, 6)),
            f_ext_follower=jnp.zeros((self.n_nodes, 6)) if use_f_ext_follower else None,
            f_ext_dead=jnp.zeros((self.n_nodes, 6)) if use_f_ext_dead else None,
            f_ext_aero=jnp.zeros((self.n_nodes, 6)) if use_f_aero else None,
            f_grav=jnp.zeros((self.n_nodes, 6)) if use_f_grav else None,
            f_int=jnp.zeros((self.n_nodes, 6)),
            f_res=jnp.zeros((self.n_nodes, 6)),
            local=True,
            prescribed_dofs=prescribed_dofs,
        )

    def calculate_varphi_from_hg(self, hg: Array) -> Array:
        r"""
        Calculate the twist vector from the reference configuration to hg
        :param hg: Deformed coordinates, [n_nodes, 4, 4]
        :return: Vector of twists, [n_nodes, 6]
        """
        return vmap(hg_to_d, (0, 0), 0)(self.hg0, hg)

    @staticmethod
    def calculate_phi_from_hg(hg_nm1: Array, hg_n: Array) -> Array:
        r"""
        Calculate the twist increment from timestep varphi to timestep varphi+1
        :param hg_nm1: Coordinates from timestep varphi-1, [n_nodes, 4, 4]
        :param hg_n: Coordinates from timestep varphi, [n_nodes, 4, 4]
        :return: Increment phi, [n_nodes, 6]
        """
        return vmap(hg_to_d, (0, 0), 0)(
            hg_nm1,
            hg_n,
        )

    @staticmethod
    def calculate_varphi_from_phi(varphi_nm1: Array, phi_n: Array) -> Array:
        r"""
        Update the varphi vector with the timestep change phi.
        :param varphi_nm1: Twists from reference to timestep varphi, [n_nodes, 6]
        :param phi_n: Twists from timestep varphi to timestep varphi+1, [n_nodes, 6]
        :return: Twists from reference to timestep varphi+1, [n_nodes, 6]
        """
        return vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)),
            (0, 0),
            0,
        )(varphi_nm1, phi_n)

    def calculate_hg_from_varphi(self, varphi: Array) -> Array:
        exp_varphi = vmap(exp_se3)(varphi)  # [n_nodes_, 4, 4]
        return jnp.einsum("ijk,ikl->ijl", self.hg0, exp_varphi)

    def assemble_matrix_from_entries(self, entries: Array) -> Array:
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

    def assemble_vector_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global vector from element entries
        :param entries: Array of element vector entries, [n_elem, 12]
        :return: System global vector, [n_dof]
        """

        vect = jnp.zeros(self.n_dof)
        vect = vect.at[self.dof_per_elem[:, :6]].add(entries[:, :6])
        return vect.at[self.dof_per_elem[:, 6:]].add(entries[:, 6:])

    def _make_load_steps_f(
            self, f: Optional[Array], weighting: Array, apply_alpha_weighting: bool
    ) -> Optional[Array]:
        r"""
        This also includes the effect of the time integrator
        :param f:
        :param weighting:
        :return:
        """
        if f is not None:
            if apply_alpha_weighting:
                f = (
                    jnp.zeros((f.shape[0], self.n_nodes, 6))
                    .at[1:, ...]
                    .set(
                        self.time_integrator.calculate_f_alpha(
                            f_nm1=f[:-1, ...], f_n=f[1:, ...]
                        )
                    )
                )  # [n_tstep, n_nodes, 6]

            f_steps = jnp.einsum("i,...->i...", weighting, f)  # [load_steps, ...]
        else:
            f_steps = None
        return f_steps

    @staticmethod
    def make_f_ext_dead_tot(
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            i_load_step: Optional[int],
            i_ts: Optional[int],
    ) -> Optional[Array]:
        if i_ts is None and i_load_step is None:
            idx = (...,)
        elif i_ts is None and i_load_step is not None:
            idx = (i_load_step, ...)
        elif i_ts is not None and i_load_step is None:
            idx = (i_ts, ...)
        else:
            idx = (i_load_step, i_ts, ...)

        if f_ext_dead is None and f_ext_aero is None:
            return None
        elif f_ext_dead is None and f_ext_aero is not None:
            return f_ext_aero[idx]
        elif f_ext_dead is not None and f_ext_aero is None:
            return f_ext_dead[idx]
        else:
            return f_ext_dead[idx] + f_ext_aero[idx]  # type: ignore

    def _make_k_t(
            self,
            d: Array,
            p_d: Array,
            eps: Array,
    ) -> Array:
        r"""
        Assemble tangent stiffness matrix as a function of the element relative configuration vectors
        :param d: Element relative configuration, [n_elem, 6]
        :return: Elementwise stiffness matrix entries, [n_elem, 12, 12]
        """
        # compute stiffness matrix entries
        return vmap(
            partial(
                _k_t_entry,
                include_geometric=self.optional_jacobians.d_f_int_d_p_d,
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

    def _make_k_t_dead(self, rmat: Array, f_ext_dead: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from dead external forces.
        :param rmat: Rotation matrices at nodes, [n_node, 3, 3]
        :param f_ext_dead: External dead forces in global reference, [n_node, 6]
        :return: Stiffness matrix contribution from dead forces, [n_node, 6, 6]
        """
        k_t = jnp.zeros((self.n_nodes, 6, 6))

        k_t = k_t.at[:, :3, 3:].set(
            -vmap(vec_to_skew)(jnp.einsum("ikj,ik->ij", rmat, f_ext_dead[:, :3]))
        )
        k_t = k_t.at[:, 3:, 3:].set(
            -vmap(vec_to_skew)(jnp.einsum("ikj,ik->ij", rmat, f_ext_dead[:, 3:]))
        )

        return k_t

    def _make_k_t_grav(self, d: Array, p_d: Array, rmat: Array, m_t: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from gravity forces.
        :param d: Element relative configuration, [n_elem, 6]
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param rmat: Nodal rotation matrices, [n_node, 3, 3]
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :return: Stiffness matrix contribution from gravity forces, [n_elem, 12, 12]
        """

        # pertubations in mass matrix integration
        g_ab = jnp.zeros(12)
        g_ab = g_ab.at[:3].set(self.gravity_vec)
        g_ab = g_ab.at[6:9].set(self.gravity_vec)
        p_d_g = jnp.einsum("ijk,k->ij", p_d, g_ab)  # [n_elem, 6

        # computes dm/dd @ p @ g_ab
        d_mg = vmap(
            lambda m_cs_, d_, ad_, l_, p_d_g_: jax.jvp(
                lambda d__: _integrate_m_l(m_cs_, d__, ad_, l_, int_order=3),
                primals=[d_],
                tangents=[p_d_g_],
            )[1],
            (0, 0, 0, 0, 0),
            0,
        )(self.m_cs, d, self.ad_inv_o0, self.l0, p_d_g)

        # pertubations in gravity direction
        # [n_nodes_, 3, 3]
        d_g_d_omega = vmap(vec_to_skew, 0, 0)(
            jnp.einsum("ikj,k->ij", rmat, self.gravity_vec)
        )

        # adding terms of [n_elem, 12, 3]
        d_mg = d_mg.at[:, :, 3:6].add(
            jnp.einsum(
                "ijk,ikl->ijl",
                m_t[:, :, :3],
                d_g_d_omega[self.connectivity[:, 0], :, :],
            )
        )
        d_mg = d_mg.at[:, :, 9:].add(
            jnp.einsum(
                "ijk,ikl->ijl",
                m_t[:, :, 6:9],
                d_g_d_omega[self.connectivity[:, 1], :, :],
            )
        )

        return d_mg

    def _make_k_t_grav_lumped(self, rmat: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from gravity forces for the lumped masses.
        :param rmat: Nodal rotation matrices, [n_node, 3, 3]
        :return: Stiffness contribution from gravity forces for lumped masses, [n_node, 6, 6]
        """
        # [n_nodes_, 3, 3]
        d_g_d_omega = vmap(vec_to_skew, 0, 0)(
            jnp.einsum("ikj,k->ij", rmat, self.gravity_vec)
        )

        return (
            jnp.zeros((self.n_nodes, 6, 6))
            .at[:, :, 3:]
            .set(jnp.einsum("ijk,ikl->ijl", self.m_lumped[:, :, 3:], d_g_d_omega))
        )

    def _make_k_t_full(
            self,
            d: Array,
            p_d: Array,
            eps: Array,
            f_ext_dead: Optional[Array],
            rmat: Array,
            m_t: Optional[Array],
    ) -> Array:
        r"""
        Compute the full tangent stiffness matrix, with contributions from stiffness, dead forces and gravity.
        :param d: Element relative configuration, [n_elem, 6]
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Strain vectors, [n_elem, 6]
        :param f_ext_dead: External dead forces in global reference, [n_node, 6]
        :param rmat: Nodal rotation matrices, [n_node, 3, 3]
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :return: Tangent stiffness matrix with all contributions, [n_dof, n_dof]
        """

        k_t = self.assemble_matrix_from_entries(self._make_k_t(d, p_d, eps))
        if f_ext_dead is not None and self.optional_jacobians.d_f_ext_dead_d_n:
            k_t += block_diag(*self._make_k_t_dead(rmat, f_ext_dead))

        if self.use_gravity and self.optional_jacobians.d_f_grav_d_n:
            if m_t is None: raise ValueError("m_t needs to be provided")
            k_t += self.assemble_matrix_from_entries(
                self._make_k_t_grav(d, p_d, rmat, m_t)
            )
            if self.use_lumped_mass:
                k_t += block_diag(*self._make_k_t_grav_lumped(rmat))
        return k_t

    def make_m_t(self, d: Array, int_order: Literal[3, 4, 5] = 3) -> Array:
        r"""
        Assemble tangent mass matrix as a function of the element relative configuration vectors. This does not include
        the lumped mass contribution.
        :param d: Element relative configuration, [n_elem, 6]
        :param int_order: Integration order for mass matrix computation
        :return: Elementwise mass matrix, [n_elem, 12, 12]
        """
        return vmap(partial(_integrate_m_l, int_order=int_order), (0, 0, 0, 0), 0)(
            self.m_cs, d, self.ad_inv_o0, self.l0
        )

    def _make_c_t(
            self,
            d: Array,
            d_dot: Array,
            v: Array,
            int_order: Literal[1, 2, 3] = 3,
    ) -> tuple[Array, Array]:
        r"""
        Assemble tangent gyroscopic matrix. This does not include the lumped mass contribution.
        :param d: Element relative configuration, [n_elem, 6]
        :param d_dot: Element relative velocity, [n_elem, 6]
        :param v: Velocities in local frames, [n_node, 6]
        :param int_order: Integration order,
        :return: Elementwise gyroscopic C_L and C_T matrices, [n_elem, 12, 12], [n_elem, 12, 12]
        """
        clt = vmap(
            partial(
                _integrate_c_t,
                int_order=int_order,
                include_q_dot=self.optional_jacobians.d_f_gyr_d_q_dot,
            ),
            (0, 0, 0, 0, 0, 0),
            0,
        )(
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
        c_t_l = vmap(_make_c_t_lumped, (0, 0), 0)(self.m_lumped, v)  # [n_node, 2, 6, 6]
        return c_t_l[:, 0, :, :], c_t_l[:, 1, :, :]

    def _make_sys_matrix(
            self,
            m_t: Array,
            c_t: Array,
            c_t_lumped: Optional[Array],
            k_t: Array,
            t_n: Array,
            ti: TimeIntregrator,
    ) -> Array:
        r"""
        Create the system matrix for the static or dynamic analysis.
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :param c_t: Disassembled system gyroscopic matrix, [n_elem, 12, 12]
        :param c_t_lumped: Disassembled system lumped gyroscopic matrix, [n_node, 6, 6]
        :param k_t: System stiffness matrix, [n_dof, n_dof]
        :param t_n: Tangent operator T(varphi), [n_nodes_, 6, 6]
        :param ti: Time integration parameters
        :return: System matrix, [n_dof, n_dof]
        """

        # note that k_t is already assembled for convenience
        k_t_tan_n = jnp.einsum(
            "ijk,jkl->ijl", k_t.reshape(self.n_dof, -1, 6), t_n
        ).reshape(self.n_dof, self.n_dof)  # [n_dof, n_dof]

        mat = (
                  self.assemble_matrix_from_entries(
                      m_t * ti.beta_prime + c_t * ti.gamma_prime
                  )
              ) + k_t_tan_n

        if self.use_lumped_mass:
            if c_t_lumped is None: raise ValueError("c_t_lumped needs to be passed")
            mat += (
                    self.m_t_lumped * ti.beta_prime
                    + block_diag(*c_t_lumped) * ti.gamma_prime
            )

        return mat

    def make_f_int(self, p_d: Array, eps: Array) -> Array:
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
    def make_f_dead_ext(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, [n_node, 6]
        :param rmat: Deformation rotation matrices, [n_node, 3, 3]
        :return: External forces, [n_node, 6]
        """

        f_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, :3])  # [n_node, 3]
        m_rot = jnp.einsum("ikj,ik->ij", rmat, f_ext[:, 3:])  # [n_node, 3]
        return jnp.concatenate((f_rot, m_rot), axis=-1)

    def split_vector_to_elements(self, vec: Array) -> Array:
        return jnp.concatenate(
            (vec[self.connectivity[:, 0], :], vec[self.connectivity[:, 1], :]), axis=-1
        )

    def _make_f_iner_gyr(
            self, m_l: Array, c_l: Array, v: Array, v_dot: Array
    ) -> tuple[Array, Array]:
        r"""
        Compute the global inertial force vector.
        :param m_l: Disassembled system mass matrix, [n_elem, 12, 12]
        :param c_l: Disassembled system gyroscopic matrix, [n_elem, 12, 12]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :return: Inertial forces, [n_elem, 12]
        """

        v_elem = self.split_vector_to_elements(v)
        v_dot_elem = jnp.concatenate(
            (v_dot[self.connectivity[:, 0], :], v_dot[self.connectivity[:, 1], :]),
            axis=-1,
        )  # [n_elem, 12]

        return -jnp.einsum("ijk,ik->ij", m_l, v_dot_elem), -jnp.einsum(
            "ijk,ik->ij", c_l, v_elem
        )  # [n_elem, 12]

    def _make_f_iner_gyr_lumped(self, c_l_lumped: Array, v: Array, v_dot: Array) -> tuple[Array, Array]:
        r"""
        Obtain the contribution to the inertial forces from the lumped masses.
        :param c_l_lumped: Gyroscopic matrix from lumped masses, [n_node, 6, 6]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :return: Inertial forces from lumped masses, [n_node, 6]
        """
        f_iner = -jnp.einsum("ijk,ik->ij", self.m_lumped, v_dot)  # [n_node, 6]
        f_gyr = - jnp.einsum("ijk,ik->ij", c_l_lumped, v)  # [n_node, 6]
        return f_iner, f_gyr

    def make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, [n_elem, 6]
        :return: Element strain vectors, [n_elem, 6]
        """

        return (d - self.d0) / self.l0[:, None]

    def make_p_d(self, d: Array) -> Array:
        r"""
        Compute the P(d) operator as a function of the element relative configuration vectors.
        :param d: Relative configuration vectors, [n_elem, 6]
        :return: P(d) operator, [n_elem, 6, 12]
        """
        return vmap(p, (0, 0), 0)(d, self.ad_inv_o0)  # [n_elem, 6, 12]

    def make_d(self, hg: Array) -> Array:
        r"""
        Compute the element relative configuration vectors from the nodal homogeneous transformation matrices
        :param hg: Nodal homogeneous transformation matrices, [n_nodes_, 4, 4]
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

    @overload
    def resolve_forces(
            self,
            hg: Array,
            dynamic: Literal[True],
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            v: Array,
            v_dot: Array,
            stop_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Array,
        Array,
        Array,
        Array,
    ]:
        ...

    @overload
    def resolve_forces(
            self,
            hg: Array,
            dynamic: Literal[False],
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            v: None,
            v_dot: None,
            stop_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Array,
        None,
        None,
        Array,
    ]:
        ...

    def resolve_forces(
            self,
            hg: Array,
            dynamic: bool,
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            v,
            v_dot,
            stop_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Array,
        Optional[Array],
        Optional[Array],
        Array,
    ]:
        r"""
        Obtain all components of the force from a final solution
        :param hg: Nodal homogeneous transformation matrices, [n_nodes_, 4, 4]
        :param dynamic: Whether to compute dynamic forces
        :param f_ext_follower: External follower forces in local reference, [n_node, 6]
        :param f_ext_dead: External dead forces in global reference, [n_node, 6]
        :param f_ext_aero: External aero forces in global reference, [n_node, 6]
        :param v: Nodal velocities in global frame, [n_node, 6]
        :param v_dot: Nodal accelerations in global frame, [n_node, 6]
        :param stop_gradients: Whether to stop computing gradients of the inertial and gyroscopic forces with respect to
        the node coordinates, as these these are small but nonzero values in practice.
        :return: Configuration vectors, strain vectors, Dead external forces, aero external forces, gravitational forces, internal forces,
        gyroscopic forces, inertial forces and residual forces
        """

        def prop_grad(x: Array) -> Array:
            return jax.lax.stop_gradient(x) if stop_gradients else x

        d = self.make_d(hg)
        eps = self.make_eps(d)
        p_d = self.make_p_d(d)

        if dynamic or self.use_gravity:
            m_t = self.make_m_t(prop_grad(d))
        else:
            m_t = None

        if dynamic:
            d_dot = self._make_d_dot(p_d, v)
            c_l = self._make_c_t(prop_grad(d), prop_grad(d_dot), v)[0]
            c_l_lumped = self._make_c_t_lumped(v)[0] if self.use_lumped_mass else None
        else:
            d_dot, c_l, c_l_lumped = None, None, None

        this_f_res = jnp.zeros((self.n_nodes, 6))

        if f_ext_dead is not None:
            this_f_ext_dead = self.make_f_dead_ext(f_ext_dead, hg[:, :3, :3])
            this_f_res += this_f_ext_dead
        else:
            this_f_ext_dead = None

        if f_ext_aero is not None:
            this_f_ext_aero = self.make_f_dead_ext(f_ext_aero, hg[:, :3, :3])
            this_f_res += this_f_ext_aero
        else:
            this_f_ext_aero = None

        if self.use_gravity:
            this_f_grav = self.assemble_vector_from_entries(
                self._make_f_grav(m_t, hg[:, :3, :3])  # type: ignore
            ).reshape(-1, 6)
            if self.use_lumped_mass:
                this_f_grav += self._make_f_grav_lumped(hg[:, :3, :3])
            this_f_res += this_f_grav
        else:
            this_f_grav = None

        this_f_int = self.assemble_vector_from_entries(
            self.make_f_int(p_d, eps)
        ).reshape(-1, 6)
        this_f_res += this_f_int

        if dynamic:
            this_f_iner, this_f_gyr = self._make_f_iner_gyr(m_t, c_l, v, v_dot)  # type: ignore
            this_f_iner = self.assemble_vector_from_entries(this_f_iner).reshape(-1, 6)
            this_f_gyr = self.assemble_vector_from_entries(this_f_gyr).reshape(-1, 6)

            if self.use_lumped_mass:
                this_f_iner += sum(self._make_f_iner_gyr_lumped(c_l_lumped, v, v_dot))  # type: ignore
            this_f_res += this_f_iner
        else:
            this_f_iner = None
            this_f_gyr = None

        if f_ext_follower is not None:
            this_f_res += f_ext_follower

        return (
            d,
            eps,
            this_f_ext_dead,
            this_f_ext_aero,
            this_f_grav,
            this_f_int,
            this_f_gyr,
            this_f_iner,
            this_f_res,
        )

    @overload
    def make_f_res(
            self,
            solve_dofs: Optional[Array],
            p_d: Array,
            eps: Array,
            hg: Array,
            f_ext_follower_n: Optional[Array],
            f_ext_dead_n: Optional[Array],
            dynamic: Literal[True],
            m_t: Array,
            c_l: Array,
            c_l_lumped: Optional[Array],
            v: Array,
            v_dot: Array,
    ) -> tuple[Array, Array]:
        ...

    @overload
    def make_f_res(
            self,
            solve_dofs: Optional[Array],
            p_d: Array,
            eps: Array,
            hg: Array,
            f_ext_follower_n: Optional[Array],
            f_ext_dead_n: Optional[Array],
            dynamic: Literal[False],
            m_t: Optional[Array],
            c_l: None,
            c_l_lumped: None,
            v: None,
            v_dot: None,
    ) -> tuple[Array, Array]:
        ...

    def make_f_res(
            self,
            solve_dofs: Optional[Array],
            p_d: Array,
            eps: Array,
            hg: Array,
            f_ext_follower_n: Optional[Array],
            f_ext_dead_n: Optional[Array],
            dynamic: bool,
            m_t,
            c_l,
            c_l_lumped,
            v,
            v_dot,
    ) -> tuple[Array, Array]:
        r"""
        Compute the residual force vector for a given configuration and external forces, used in the nonlinear solve.
        This is the force imbalance that the nonlinear solver will seek to drive to zero. Additionally, returns an
        "absolute sum" of all forces, used for relative convergence checks.
        :param solve_dofs: Optional array of degrees of freedom to solve for [n_solve_dofs]
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :param hg: Nodal homogeneous transformation matrices, [n_nodes_, 4, 4]
        :param f_ext_follower_n: Nodal follower forces, [n_node, 6]
        :param f_ext_dead_n: Nodal dead forces, [n_node, 6]
        :param dynamic: Flag for whether to compute dynamic entries
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12]
        :param c_l: Dissembled system gyroscopic matrix, [n_elem, 12, 12]
        :param c_l_lumped: Lumped gyroscopic matrix, [n_node, 6, 6]
        :param v: Nodal velocities, [n_node, 6]
        :param v_dot: Nodal accelerations, [n_node, 6]
        :return: Residual force vector, [n_dof], absolute sum of forces, [n_dof]
        """

        f_res = self.make_f_int(p_d, eps)  # [n_elem, 12]
        f_abs_sum = jnp.abs(f_res)

        if self.use_gravity:
            f_grav = self._make_f_grav(m_t, hg[:, :3, :3])
            f_res += f_grav
            f_abs_sum += jnp.abs(f_grav)

        if dynamic:
            f_iner, f_gyr = self._make_f_iner_gyr(m_t, c_l, v, v_dot)
            f_res += f_iner + f_gyr
            f_abs_sum += jnp.abs(f_iner + f_gyr)

        f_res_vect = self.assemble_vector_from_entries(f_res)
        f_abs_sum_vect = self.assemble_vector_from_entries(f_abs_sum)

        if f_ext_follower_n is not None:
            f_res_vect += f_ext_follower_n.reshape(self.n_dof).ravel()
            f_abs_sum_vect += jnp.abs(f_ext_follower_n.reshape(self.n_dof).ravel())
        if f_ext_dead_n is not None:
            f_dead = self.make_f_dead_ext(f_ext_dead_n, hg[:, :3, :3]).ravel()
            f_res_vect += f_dead
            f_abs_sum_vect += jnp.abs(f_dead)

        if self.use_lumped_mass:
            if dynamic:
                f_iner_gyr_lumped = jnp.add(*self._make_f_iner_gyr_lumped(c_l_lumped, v, v_dot)).ravel()

                f_res_vect += f_iner_gyr_lumped
                f_abs_sum_vect += jnp.abs(f_iner_gyr_lumped)
            if self.use_gravity:
                f_grav_lumped = self._make_f_grav_lumped(hg[:, :3, :3]).ravel()
                f_res_vect += f_grav_lumped
                f_abs_sum_vect += jnp.abs(f_grav_lumped)

        if solve_dofs is not None:
            return f_res_vect[solve_dofs], f_abs_sum_vect[
                solve_dofs
            ]  # [n_solve_dof], [n_solve_dof]
        else:
            return f_res_vect, f_abs_sum_vect  # [n_dof], [n_dof]

    @staticmethod
    def update_hg(hg: Array, phi: Array) -> Array:
        r"""
        Update the nodal homogeneous transformation matrices with the configuration increments.
        :param hg: Existing nodal homogeneous transformation matrices, [n_nodes_, 4, 4]
        :param phi: Perturbation to the configuration vector, [n_nodes_, 6]
        :return: Updated nodal homogeneous transformation matrices, [n_nodes_, 4, 4]
        """
        return jnp.einsum(
            "ijk,ikl->ijl",
            hg,
            vmap(exp_se3, 0, 0)(phi.reshape(-1, 6)),
        )

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
            f_ext_aero: Optional[Array],
            prescribed_dofs: Sequence[int] | Array | slice | int,
            load_steps: int = 1,
            relaxation_factor: float = 1.0,
    ) -> StaticStructure:
        r"""
        Perform static solve of the structure_dv under external loads
        :param f_ext_follower: External forces array of follower forces [n_node, 6]
        :param f_ext_dead: External forces array of dead loads [n_node, 6]
        :param f_ext_aero: External forces array of aerodynamic loads [n_node, 6]
        :param prescribed_dofs: Index of degrees of freedom which are prescribed (not solved for).
        :param load_steps: Number of load steps to apply the external loads over.
        :param relaxation_factor: Relaxation factor for updates, in range (0, 1].
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
        solve_dofs = jnp.setdiff1d(
            jnp.arange(self.n_dof),
            prescribed_dofs_arr,
            size=self.n_dof - prescribed_dofs_arr.size,
        )

        # process external forces for load stepping
        load_step_weight: Array = jnp.linspace(0.0, 1.0, load_steps + 1)[
            1:
        ]  # [load_steps]

        f_ext_follower_steps = self._make_load_steps_f(
            f_ext_follower, load_step_weight, apply_alpha_weighting=False
        )
        f_ext_dead_steps = self._make_load_steps_f(
            f_ext_dead, load_step_weight, apply_alpha_weighting=False
        )
        f_ext_aero_steps = self._make_load_steps_f(
            f_ext_aero, load_step_weight, apply_alpha_weighting=False
        )

        def _update(
                i_load_step: int,
                converge_status: ConvergenceStatus,
                hg_n: Array,
        ) -> tuple[int, ConvergenceStatus, Array]:
            # base parameters
            d_n = self.make_d(hg_n)  # [n_elem, 6]
            p_d_n = self.make_p_d(d_n)  # [n_elem, 6, 12]
            eps_n = self.make_eps(d_n)  # [n_elem, 6]
            m_t = self.make_m_t(d_n) if self.use_gravity else None  # [n_elem, 12, 12]

            # get total dead forces for this load step, [n_node, 6]
            total_f_ext_dead_step = self.make_f_ext_dead_tot(
                f_ext_dead_steps, f_ext_aero_steps, i_load_step, i_ts=None
            )

            # assemble tangent stiffness matrix, [n_solve_dof, n_solve_dof]
            k_t_solve_n = self._make_k_t_full(
                d_n,
                p_d_n,
                eps_n,
                total_f_ext_dead_step,
                hg_n[:, :3, :3],
                m_t,
            )[jnp.ix_(solve_dofs, solve_dofs)]

            # compute residual forces, [n_solve_dofs]
            f_res_solve_n, f_abs_sum_n = self.make_f_res(
                solve_dofs,
                p_d_n,
                eps_n,
                hg_n,
                f_ext_follower_steps[i_load_step, ...]
                if f_ext_follower_steps is not None
                else None,
                total_f_ext_dead_step,
                False,
                m_t,
                None,
                None,
                None,
                None,
            )

            # solve for configuration increment, [n_solve_dofs]
            d_varphi_np1 = (
                    jnp.linalg.solve(k_t_solve_n, f_res_solve_n) * relaxation_factor
            )

            # update configuration, [n_nodes_, 4, 4]
            hg_np1_full = self.update_hg(
                hg_n, jnp.zeros(self.n_dof).at[solve_dofs].set(d_varphi_np1)
            )

            # algebra between undeformed and deformed arr_list_shapes, used to check relative convergence, [n_solve_dofs]
            if self.convergence_settings.rel_disp_tol is not None:
                # TODO: this is relative expensive to compute
                h_full = vmap(hg_to_d, (0, 0), 0)(self.hg0, hg_np1_full).ravel()[
                    solve_dofs
                ]
            else:
                h_full = None

            # update convergence status
            converge_status.update(
                delta_disp=d_varphi_np1,
                total_disp=h_full,
                delta_force=f_res_solve_n,
                total_force=f_abs_sum_n,
            )

            if self.verbosity.value == VerbosityLevel.VERBOSE.value:
                converge_status.print_struct_message(None, i_load_step)

            return i_load_step, converge_status, hg_np1_full

        def convergence_loop(
                i_load_step: int,
                hg_init: Array,
        ) -> Array:
            r"""
            Convergence loop
            :param i_load_step:
            :param hg_init:
            :return:
            """
            _, convergence_status, hg_solve = jax.lax.while_loop(
                lambda args_: ~args_[1].get_status(),
                lambda args_: _update(*args_),
                (
                    i_load_step,
                    ConvergenceStatus(
                        self.convergence_settings,
                    ),
                    hg_init,
                ),
            )

            if self.verbosity == VerbosityLevel.NORMAL:
                convergence_status.print_struct_message(None, i_load_step)

            return hg_solve

        # solve for each load step
        hg = jax.lax.fori_loop(
            0,
            load_steps,
            lambda *args: convergence_loop(*args),
            self.hg0,
        )

        # postprocess final results
        d, eps, f_ext_dead_local, f_ext_aero_local, f_grav, f_int, _, _, f_res = (
            self.resolve_forces(
                hg=hg,
                dynamic=False,
                f_ext_dead=f_ext_dead,
                f_ext_follower=f_ext_follower,
                f_ext_aero=f_ext_aero,
                v=None,
                v_dot=None,
            )
        )
        varphi = self.calculate_varphi_from_hg(hg)

        return StaticStructure(
            hg=hg,
            conn=self.connectivity,
            o0=self.o0,
            d=d,
            eps=eps,
            varphi=varphi,
            f_int=f_int,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead_local,
            f_ext_aero=f_ext_aero_local,
            f_grav=f_grav,
            f_res=f_res,
            prescribed_dofs=prescribed_dofs_arr,
        )

    def dynamic_solve(
            self,
            init_state: Optional[DynamicStructureSnapshot | StaticStructure],
            n_tstep: int,
            dt: Array | float,
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            prescribed_dofs: Sequence[int] | Array | slice | int | None,
            load_steps: int = 1,
            relaxation_factor: float = 1.0,
            spectral_radius: float = 1.0,
    ) -> DynamicStructure:
        r"""
        Perform dynamic solve of the structure_dv under external loads
        :param init_state: Initial state of the structure_dv, either as a DynamicStructureSnapshot or StaticStructure. If
        None, the reference configuration is used with zero velocities.
        :param n_tstep: Number of time steps to simulate
        :param dt: Time step length
        :param f_ext_follower: Following external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external follower forces
        :param f_ext_dead: Dead external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external dead forces
        :param f_ext_aero: Aerodynamic external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external aerodynamic forces
        :param prescribed_dofs: Degrees of freedom which are prescribed (not solved for).
        :param load_steps: Number of load steps to apply the external loads over.
        :param relaxation_factor: Relaxation factor for Newton-Raphson iterations, in the range (0, 1].
        :param spectral_radius: Spectral radius for the time integrator, in the range [0, 1].
        :return: DynamicStructure dataclass containing results of the dynamic analysis.
        """

        # add a warning if using 32-bit floats
        warn_if_32_bit()

        # set up initial state
        if init_state is None:
            init_state_: DynamicStructureSnapshot = self.reference_configuration(
                use_f_aero=f_ext_aero is not None,
                use_f_ext_dead=f_ext_dead is not None,
                use_f_ext_follower=f_ext_follower is not None,
            ).to_dynamic()
        elif isinstance(init_state, StaticStructure):
            init_state_ = init_state.to_dynamic()
        else:
            init_state_ = init_state

        if not (0.0 < relaxation_factor <= 1.0):
            raise ValueError("relaxation_factor must be in the range (0, 1]")

        if load_steps <= 0:
            raise ValueError("load_steps must be a positive integer")

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = jnp.setdiff1d(jnp.arange(self.n_dof), prescribed_dofs_arr)

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
        f_ext_aero = check_force(f_ext_aero, "f_ext_aero")  # [n_tstep, n_node, 6]

        # time integration parameters
        dt: Array = jnp.array(dt)
        self.time_integrator = TimeIntregrator(spectral_radius=spectral_radius, dt=dt)

        # process external forces for load stepping
        load_step_weight: Array = jnp.linspace(0.0, 1.0, load_steps + 1)[
            1:
        ]  # [load_steps]
        f_ext_follower_alpha_steps = self._make_load_steps_f(
            f_ext_follower, load_step_weight, apply_alpha_weighting=True
        )
        f_ext_dead_alpha_steps = self._make_load_steps_f(
            f_ext_dead, load_step_weight, apply_alpha_weighting=True
        )
        f_ext_aero_alpha_steps = self._make_load_steps_f(
            f_ext_aero, load_step_weight, apply_alpha_weighting=True
        )

        def _update(
                i_load_step: int,
                i_ts: int,
                converge_status_: ConvergenceStatus,
                hg_n: Array,
                phi_alpha: Array,
                q_alpha: StructureMinimalStates,
        ) -> tuple[
            int,
            int,
            ConvergenceStatus,
            Array,
            Array,
            StructureMinimalStates,
        ]:
            r"""
            Solution update for a single iteration of the nonlinear solver at a given time step and load step
            :param i_load_step: Loadstep index
            :param i_ts: Time step index
            :param converge_status_: ConvergenceStatus object for the current iteration, used to track convergence and
            print messages.
            :param hg_n: Transformation matrices at iteration varphi, [n_nodes_, 4, 4]
            :return: Load and time step indices, updated ConvergenceStatus object, updated transformation matrices,
            configuration, velocities and accelerations for iteration varphi+1.
            """

            hg_update = self.update_hg(hg_n, phi_alpha)  # [n_node, 4, 4]

            # base parameters
            d_n = self.make_d(hg_update)  # [n_elem, 6]
            p_d_n = self.make_p_d(d_n)  # [n_elem, 6, 12]
            eps_n = self.make_eps(d_n)  # [n_elem, 6]
            d_dot_n = self._make_d_dot(p_d_n, q_alpha.v)  # [n_elem, 6]
            t_n = vmap(t_se3, 0, 0)(phi_alpha)  # [n_node, 6, 6]

            # tangent matrices
            m_t = self.make_m_t(d_n)  # [n_elem, 12, 12]
            c_l, c_t = self._make_c_t(
                d_n, d_dot_n, q_alpha.v
            )  # [n_elem, 12, 12], [n_elem, 12, 12]

            total_f_ext_dead = self.make_f_ext_dead_tot(
                f_ext_dead_alpha_steps, f_ext_aero_alpha_steps, i_load_step, i_ts
            )  # [n_node, 6]

            k_t = self._make_k_t_full(
                d_n,
                p_d_n,
                eps_n,
                total_f_ext_dead,
                hg_n[:, :3, :3],
                m_t,
            )  # [n_dof, n_dof]

            # add lumped mass contributions if applicable
            if self.use_lumped_mass:
                c_l_lumped, c_t_lumped = self._make_c_t_lumped(
                    q_alpha.v
                )  # [n_node, 6, 6], [n_node, 6, 6]
            else:
                c_l_lumped, c_t_lumped = None, None

            # residual forces, [n_solve_dofs]
            f_res_n_solve, f_abs_sum_n = self.make_f_res(
                solve_dofs,
                p_d_n,
                eps_n,
                hg_n,
                f_ext_follower_alpha_steps[i_load_step, i_ts, ...]
                if f_ext_follower_alpha_steps is not None
                else None,
                total_f_ext_dead,
                True,
                m_t,
                c_l,
                c_l_lumped,
                q_alpha.v,
                q_alpha.v_dot,
            )

            # system matrix, [n_solve_dofs, n_solve_dofs]
            sys_mat = self._make_sys_matrix(
                m_t=m_t,
                c_t=c_t,
                c_t_lumped=c_t_lumped,
                k_t=k_t,
                t_n=t_n,
                ti=self.time_integrator,
            )[jnp.ix_(solve_dofs, solve_dofs)]

            # solve for configuration increment, [n_solve_dofs]
            d_n_np1 = jnp.linalg.solve(sys_mat, f_res_n_solve) * relaxation_factor
            phi_np1 = phi_alpha.ravel().at[solve_dofs].add(d_n_np1).reshape(-1, 6)

            # update configuration, velocities and accelerations
            v_np1 = (
                q_alpha.v.ravel()
                .at[solve_dofs]
                .add(self.time_integrator.gamma_prime * d_n_np1)
                .reshape(-1, 6)
            )
            v_dot_np1 = (
                q_alpha.v_dot.ravel()
                .at[solve_dofs]
                .add(self.time_integrator.beta_prime * d_n_np1)
                .reshape(-1, 6)
            )

            # update convergence status
            converge_status_.update(
                delta_disp=d_n_np1,
                total_disp=phi_np1,
                delta_force=f_res_n_solve,
                total_force=f_abs_sum_n,
            )

            if self.verbosity.value == VerbosityLevel.VERBOSE.value:
                converge_status_.print_struct_message(t[i_ts], i_load_step)

            q_alpha_update = StructureMinimalStates(
                varphi=None, v=v_np1, v_dot=v_dot_np1, a=q_alpha.a
            )

            return i_load_step, i_ts, converge_status_, hg_n, phi_np1, q_alpha_update

        def time_step_loop(
                i_ts: int, sol: DynamicStructure, converge_status_: ConvergenceStatus
        ) -> tuple[DynamicStructure, ConvergenceStatus]:
            r"""
            Performs analysis on a single time step, including load stepping
            :param i_ts: Index of time step to solve
            :param sol: Solution object, with results up to time step i_ts-1.
            :param converge_status_: Convergence status object.
            :return: Solution object with results up to time step i_ts.
            """

            # predictor step
            phi_init, q_init = self.time_integrator.predict_q(sol.get_minimal_states(i_ts - 1))
            phi_alpha_init, q_alpha_init = self.time_integrator.calculate_q_alpha(
                q_nm1=sol.get_minimal_states(i_ts - 1), q_n=q_init, phi_n=phi_init
            )

            q_alpha_init.varphi = None  # this value is not used during the loop

            # solve
            _, converge_status_, hg, phi_alpha, q_alpha = load_step_loop(
                i_ts,
                converge_status_,
                sol.hg[i_ts - 1, ...],
                phi_alpha_init,
                q_alpha_init,
            )

            # postprocess results for time step and store in solution object
            q_n, phi_n = self.time_integrator.calculate_q_n_from_q_alpha(
                q_alpha=q_alpha,
                q_nm1=sol.get_minimal_states(i_ts - 1),
                phi_alpha=phi_alpha,
            )

            # update pseudoacceleration
            q_n.a = self.time_integrator.calculate_a_n(
                a_nm1=sol.a[i_ts - 1, ...],
                v_dot_nm1=sol.v_dot[i_ts - 1, ...],
                v_dot_n=q_n.v_dot,
            )

            hg_n = self.update_hg(sol.hg[i_ts - 1, ...], phi_n)
            (
                d,
                eps,
                f_ext_dead_local,
                f_ext_aero_local,
                f_grav,
                f_int,
                f_gyr,
                f_iner,
                f_res,
            ) = self.resolve_forces(
                hg=hg_n,
                dynamic=True,
                f_ext_dead=f_ext_dead[i_ts, ...] if f_ext_dead is not None else None,
                f_ext_follower=f_ext_follower[i_ts, ...]
                if f_ext_follower is not None
                else None,
                f_ext_aero=f_ext_aero[i_ts, ...] if f_ext_aero is not None else None,
                v=q_n.v,
                v_dot=q_n.v_dot,
            )
            sol.d = sol.d.at[i_ts, ...].set(d)
            sol.eps = sol.eps.at[i_ts, ...].set(eps)
            sol.v = sol.v.at[i_ts, ...].set(q_n.v)
            sol.v_dot = sol.v_dot.at[i_ts, ...].set(q_n.v_dot)
            sol.a = sol.a.at[i_ts, ...].set(q_n.a)
            sol.hg = sol.hg.at[i_ts, ...].set(hg_n)
            if f_ext_follower is not None:
                sol.f_ext_follower = sol.f_ext_follower.at[i_ts, ...].set(
                    f_ext_follower[i_ts, ...]
                )
            if f_ext_dead is not None:
                sol.f_ext_dead = sol.f_ext_dead.at[i_ts, ...].set(f_ext_dead_local)
            if f_ext_aero is not None:
                sol.f_ext_aero = sol.f_ext_aero.at[i_ts, ...].set(f_ext_aero_local)
            if self.use_gravity:
                if sol.f_grav is None: raise ValueError("sol.f_grav is None")
                sol.f_grav = sol.f_grav.at[i_ts, ...].set(f_grav)
            sol.f_int = sol.f_int.at[i_ts, ...].set(f_int)
            sol.f_iner_gyr = sol.f_iner_gyr.at[i_ts, ...].set(f_iner + f_gyr)
            sol.f_res = sol.f_res.at[i_ts, ...].set(f_res)

            return sol, converge_status_

        def convergence_loop(
                i_load_step: int,
                i_ts: int,
                converge_status_: ConvergenceStatus,
                hg_n: Array,
                phi_n: Array,
                q_n: StructureMinimalStates,
        ) -> tuple[int, ConvergenceStatus, Array, Array, StructureMinimalStates]:
            r"""
            Convergence loop within each load step of a time step.
            :param i_load_step: Load step index
            :param i_ts: Time step index
            :param converge_status_: ConvergenceStatus object to update with convergence information during load
            stepping.
            :param hg_n: Node transformations at the beginning of the load step, [n_nodes_, 4, 4]
            :param phi_n: Node configuration increments in algebra space, [n_nodes_, 6]
            :param q_n: Minimal states
            :return: Time step index, convergence status, and updated configuration, velocities and accelerations.
            """

            converge_status_.reset_status()

            _, _, converge_status_, hg_solve, phi_n, q_n = jax.lax.while_loop(
                lambda args_: ~args_[2].get_status(),
                lambda args_: _update(*args_),
                (i_load_step, i_ts, converge_status_, hg_n, phi_n, q_n),
            )

            if self.verbosity.value == VerbosityLevel.NORMAL.value:
                converge_status_.print_struct_message(t[i_ts], i_load_step)

            return i_ts, converge_status_, hg_solve, phi_n, q_n

        def load_step_loop(
                i_ts: int,
                converge_status_: ConvergenceStatus,
                hg_n: Array,
                phi_n: Array,
                q_n: StructureMinimalStates,
        ) -> tuple[int, ConvergenceStatus, Array, Array, StructureMinimalStates]:
            r"""
            Performs load stepping iterations for a given time step
            :param i_ts: Timestep index for which to perform load stepping
            :param converge_status_: ConvergenceStatus object to update with load stepping convergence information
            :param hg_n: SE(3) nodal transformation matrices at the beginning of the load step, [n_nodes_, 4, 4]
            :param phi_n: Nodal updates to the configuration in the algebra space, [n_nodes_, 6]
            :param q_n: Minimal states
            :return: Time step index, updated ConvergenceStatus object, and updated configuration, velocities and accelerations after load stepping
            """
            return jax.lax.fori_loop(
                0,
                load_steps,
                lambda i_load_step, args: convergence_loop(i_load_step, *args),
                (i_ts, converge_status_, hg_n, phi_n, q_n),
            )

        def evaluate_initial_equilibrium(
                init_state__: DynamicStructureSnapshot,
        ) -> DynamicStructureSnapshot:
            r"""
            Evaluates the forces for a given initial state to check whether it is in equilibrium. If not, a warning is
            raised with the maximum residual force. This is important to ensure that the time integration starts from a
            consistent state.
            :param init_state__: DynamicStructureSnapshot containing the initial state to evaluate.
            :return: DynamicStructureSnapshot with the forces evaluated for the initial state.
            """
            d, eps, f_ext_dead_, f_ext_aero_, f_grav, f_int, f_gyr, f_iner, f_res = (
                self.resolve_forces(
                    hg=init_state__.hg,
                    dynamic=True,
                    f_ext_dead=init_state__.f_ext_dead,
                    f_ext_aero=init_state__.f_ext_aero,
                    f_ext_follower=init_state__.f_ext_follower,
                    v=init_state__.v,
                    v_dot=init_state__.v_dot,
                )
            )

            max_res = jnp.max(jnp.abs(f_res))
            if max_res > 1e-6:
                warn(
                    "Initial state is not in equilibrium, maximum residual force is {max_res:.3e}",
                    max_res=max_res,
                )

            return DynamicStructureSnapshot(
                hg=init_state__.hg,
                conn=self.connectivity,
                o0=self.o0,
                d=d,
                eps=eps,
                varphi=init_state__.varphi,
                v=init_state__.v,
                v_dot=init_state__.v_dot,
                a=init_state__.v_dot,  # initial pseudoacceleration set equal to initial acceleration
                f_ext_follower=init_state__.f_ext_follower,
                f_ext_dead=f_ext_dead_,
                f_ext_aero=f_ext_aero_,
                f_grav=f_grav,
                f_int=f_int,
                f_iner_gyr=f_iner + f_gyr,  # type: ignore
                f_res=f_res,
                t=init_state__.t,
                i_ts=init_state__.i_ts,
                prescribed_dofs=prescribed_dofs_arr,
            )

        # initialise problem
        t = jnp.arange(n_tstep) * dt + init_state_.t
        init_state_eval = evaluate_initial_equilibrium(init_state_)
        init_dynamic_state = DynamicStructure.initialise(
            initial_snapshot=init_state_eval, t=t, prescribed_dofs=prescribed_dofs_arr
        )
        converge_status = ConvergenceStatus(
            convergence_settings=self.convergence_settings
        )

        # solve
        output, _ = jax.lax.fori_loop(
            1,
            n_tstep,
            lambda i_ts, args: time_step_loop(i_ts, *args),
            (init_dynamic_state, converge_status),
        )

        return output
