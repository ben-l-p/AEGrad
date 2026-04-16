from __future__ import annotations

from typing import Optional, Sequence, Literal, overload, TYPE_CHECKING, cast
from functools import partial

from jax import numpy as jnp
from jax import Array, vmap
import jax
from jax.scipy.linalg import block_diag

from aegrad.aero.data_structures import DynamicAeroCase
from aegrad.utils.utils import check_type
from aegrad.structure.data_structures import (
    StaticStructure,
    DynamicStructure,
    DynamicStructureSnapshot,
    OptionalJacobians,
)
from aegrad.utils.data_structures import ConvergenceSettings, ConvergenceStatus
from aegrad.utils.print_utils import warn, warn_if_32_bit, VerbosityLevel, VERBOSITY_LEVEL
from aegrad.structure.utils import _check_connectivity, _n_elem_per_node, get_solve_dofs, transform_nodal_vect
from aegrad.algebra.array_utils import check_arr_shape, check_arr_dtype
from aegrad.structure.utils import (
    _k_t_entry,
    _integrate_m_l,
    _integrate_c_t,
    _make_c_t_lumped,
)
from aegrad.algebra.se3 import p, rmat_to_ha_hat, hg_to_d, exp_se3, ha_to_ha_tilde
from aegrad.algebra.so3 import vec_to_skew
from aegrad.structure.time_integration import TimeIntegrator
from aegrad.algebra.se3 import t_se3
from aegrad.structure.gradients.data_structures import StructuralDesignVariables, StructureFullStates
from aegrad.structure.data_structures import StructureMinimalStates

if TYPE_CHECKING:
    from aegrad.coupled.data_structures import DynamicAeroelastic
    from aegrad.aero.uvlm import UVLM


class BaseBeamStructure:
    r"""
    Class to represent nonlinear beam structure model
    """

    def __init__(
            self,
            num_nodes: int,
            connectivity: Array,
            y_vector: Array,
            k_cs_index: Optional[Array] = None,
            m_cs_index: Optional[Array] = None,
            m_lumped_index: Optional[Array] = None,
            gravity: Optional[Array] = None,
            optional_jacobians: Optional[OptionalJacobians] = None,
            struct_convergence_settings: ConvergenceSettings = ConvergenceSettings(max_n_iter=25,
                                                                                   rel_disp_tol=1e-6,
                                                                                   abs_disp_tol=1e-8,
                                                                                   rel_force_tol=1e-6,
                                                                                   abs_force_tol=1e-8)
    ) -> None:
        r"""
        Initialise BaseBeamStructure class with all non-design parameters.
        :param num_nodes: Number of nodes in the structure.
        :param connectivity: Connectivity array of arr_list_shapes [n_elem, 2].
        :param y_vector: Vector defining the y direction for each element, [n_elem, 3].
        :param k_cs_index: Array defining the index from the library of k_cs to use for each element, [n_elem]. If none
        is passed, it is assumed that there is a single entry in the library which is used for all elements.
        :param m_cs_index: Array defining the index from the library of m_cs to use for each element, [n_elem]. If none
        is passed, it is assumed that there is a single entry in the library which is used for all elements.
        :param m_lumped_index: Node index for nodes which are to have a lumped mass attached. The order is the same as
        that for the lumped mass data [n_lumped_mass].
        :param gravity: Gravity vector in global reference frame, or None for no gravity_vec, [3].
        :param optional_jacobians: Define which Jacobians contributions are to be used for solution.
        :param struct_convergence_settings: Structure convergence settings.
        """

        check_type(num_nodes, int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        check_arr_shape(connectivity, (None, 2), "connectivity")
        check_arr_dtype(connectivity, int, "connectivity")
        _check_connectivity(connectivity, num_nodes)
        self.connectivity: Array = connectivity  # [n_elem, 2]
        self.n_elem_per_node: Array = _n_elem_per_node(connectivity)  # [n_nodes]
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
        self.use_lumped_mass: bool = m_lumped_index is not None
        # initialise design variables with default values
        self.x0: Array = jnp.zeros((num_nodes, 3))

        self._m_cs: Optional[Array] = None
        self._k_cs: Optional[Array] = None
        self._m_lumped: Optional[Array] = None

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

        # indexing
        if k_cs_index is None:
            k_cs_index_ = jnp.zeros(self.n_elem, dtype=int)
        else:
            check_arr_shape(k_cs_index, (self.n_elem,), "k_cs_index")
            check_arr_dtype(k_cs_index, int, "k_cs_index")
            k_cs_index_ = k_cs_index
        self.k_cs_index: Array = k_cs_index_

        if m_cs_index is None:
            m_cs_index_ = jnp.zeros(self.n_elem, dtype=int)
        else:
            check_arr_shape(m_cs_index, (self.n_elem,), "m_cs_index")
            check_arr_dtype(m_cs_index, int, "k_cs_index")
            m_cs_index_ = m_cs_index
        self.m_cs_index: Array = m_cs_index_

        if m_lumped_index is not None:
            check_arr_dtype(m_lumped_index, int, "m_lumped_index")
        self.m_lumped_index: Optional[Array] = m_lumped_index

        # other settings
        self.use_m_cs: bool = False

        self.optional_jacobians: OptionalJacobians = (
            optional_jacobians
            if optional_jacobians is not None
            else OptionalJacobians()
        )
        self.struct_convergence_settings: ConvergenceSettings = struct_convergence_settings

        self._time_integrator: Optional[TimeIntegrator] = None

    @property
    def k_cs(self) -> Array:
        if self._k_cs is None:
            raise ValueError("k_cs has not been set. Please set k_cs before accessing.")
        return self._k_cs

    @k_cs.setter
    def k_cs(self, k_cs: Array) -> None:
        self._k_cs = k_cs

    @property
    def m_cs(self) -> Array:
        if self._m_cs is None:
            raise ValueError("m_cs has not been set. Please set m_cs before accessing.")
        return self._m_cs

    @m_cs.setter
    def m_cs(self, m_cs: Array) -> None:
        self._m_cs = m_cs

    @property
    def m_lumped(self) -> Array:
        if self._m_lumped is None:
            raise ValueError("m_lumped has not been set. Please set m_lumped before accessing.")
        return self._m_lumped

    @m_lumped.setter
    def m_lumped(self, m_lumped: Array) -> None:
        self._m_lumped = m_lumped

    @property
    def time_integrator(self) -> TimeIntegrator:
        if self._time_integrator is None:
            raise ValueError(
                "Time integrator has not been set. Please set time_integrator before accessing."
            )
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, ti: TimeIntegrator) -> None:
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
        Set design variables and compute initial configuration dependent quantities.
        :param coords: Node coordinates, [n_nodes, 3].
        :param k_cs: Cross-section stiffness matrices, [n_entry, 6, 6] or [6, 6].
        :param m_cs: Cross-section mass matrices, [n_entry, 6, 6] or [6, 6].
        :param m_lumped: Lumped mass matrices at nodes, [n_entry, 6, 6].
        :param remove_checks: Flag to ignore input checks, used when function is JIT compiled.
        """

        # coordinates
        check_arr_shape(coords, (self.n_nodes, 3), "coords")
        self.x0 = self.x0.at[...].set(coords)

        # populate arrays
        if k_cs.ndim == 2:
            k_cs = k_cs[None, ...]
        check_arr_shape(k_cs, (None, 6, 6), "k_cs")

        if not remove_checks and k_cs.shape[0] != jnp.unique_values(self.k_cs_index).size:
            warn(
                "Redundant values in k_cs which are not used for solution due to no corresponding entry in k_cs_index.")

        self.k_cs = k_cs
        if m_cs is None:
            if not remove_checks and self.use_gravity and m_lumped is None:
                warn(
                    "No mass matrices provided, but gravity is enabled. Assuming zero mass.",
                )
            m_cs_ = jnp.zeros((6, 6))
        else:
            m_cs_ = m_cs

        if m_cs_.ndim == 2:
            m_cs_ = m_cs_[None, ...]

        check_arr_shape(m_cs_, (None, 6, 6), "m_cs")

        if not remove_checks and m_cs_.shape[0] != jnp.unique_values(self.m_cs_index).size and m_cs is not None:
            warn(
                "Redundant values in m_cs which are not used for solution due to no corresponding entry in "
                "m_cs_index.")

        self.m_cs = m_cs_

        if m_lumped is not None:
            if not remove_checks:
                check_arr_shape(m_lumped, (None, 6, 6), "m_lumped")

                if self.m_lumped_index is None:
                    raise ValueError("m_lumped_index has not been set")

                if m_lumped.shape[0] != self.m_lumped_index.size:
                    raise ValueError(
                        "Number of entries in m_lumped does not match number of indices in m_lumped_index.")

            self.m_lumped = m_lumped

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
        )  # [n_nodes, 4, 4]
        self.hg0 = self.hg0.at[:, :3, 3].set(self.x0)

    def get_design_variables(self, struct_case: StaticStructure | DynamicStructure) -> StructuralDesignVariables:
        r"""
        Obtain the design variables for the structural problem. As the external forcing is defined for each solve, the
        chosen forcing is required as input.
        :param struct_case: Structural case
        :return: StructuralDesignVariables dataclass containing design variables
        """

        # struct_case.f_ext_dead is stored in local frame: f_local = R^T @ f_global,
        # so recover f_global = R @ f_local
        hg = struct_case.hg
        if hg.ndim == 4:  # DynamicStructure: [n_tstep, n_nodes, 4, 4]
            rmat = hg[:, :, :3, :3]
        else:  # StaticStructure: [n_nodes, 4, 4]
            rmat = hg[:, :3, :3]
        f_ext_dead_global = (
            transform_nodal_vect(struct_case.f_ext_dead, rmat)
            if struct_case.f_ext_dead is not None else None
        )
        return StructuralDesignVariables(x0=self.x0, m_cs=self.m_cs, k_cs=self.k_cs, m_lumped=self._m_lumped,
                                         f_ext_dead=f_ext_dead_global, f_ext_follower=struct_case.f_ext_follower,
                                         f_shape=())

    def reference_configuration(
            self,
            use_f_ext_follower: bool = True,
            use_f_ext_dead: bool = True,
            use_f_aero: bool = True,
            use_f_grav: bool = True,
            prescribed_dofs: Optional[Array] = None,
    ) -> StaticStructure:
        r"""
        Get the reference configuration of the structure.
        :return: StaticStructure dataclass containing reference configuration.
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
            f_elem=jnp.zeros((self.n_elem, 6)),
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

    def calculate_hg_from_varphi(self, varphi: Array) -> Array:
        exp_varphi = vmap(exp_se3)(varphi)  # [n_nodes, 4, 4]
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

    def add_lumped_contributions_to_arr(self, arr: Array, lumped_arr: Array) -> Array:
        r"""
        Add lumped contributions to an array
        :param arr: Full array, [6*n_node, 6*n_node]
        :param lumped_arr: Lumped contributions, [n_lump, 6, 6]
        :return: In-place updated array, [6*n_node, 6*n_node]
        """

        if self.m_lumped_index is None:
            raise ValueError("m_lumped_index is None")

        def add_block(carry, x):
            node_idx, block = x
            dofs = node_idx * 6 + jnp.arange(6)
            return carry.at[jnp.ix_(dofs, dofs)].add(block), None

        arr, _ = jax.lax.scan(add_block, arr, (self.m_lumped_index, lumped_arr))
        return arr

    def add_lumped_contributions_to_vec(self, vec: Array, lumped_vec: Array) -> Array:
        r"""
        Add lumped contributions to an array
        :param vec: Full vector, [6*n_node]
        :param lumped_vec: Lumped contributions, [n_lump, 6]
        :return: In-place updated vector, [6*n_node]
        """

        if self.m_lumped_index is None:
            raise ValueError("m_lumped_index is None")

        idx = (self.m_lumped_index[:, None] * 6 + jnp.arange(6)[None, :]).ravel()  # [n_lump * 6]

        return vec.at[idx].add(lumped_vec)

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
    ) -> Optional[Array]:

        idx = (i_load_step, ...) if i_load_step is not None else (...,)

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
            self.k_cs[self.k_cs_index, ...],
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

        # perturbations in mass matrix integration
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
        )(self.m_cs[self.m_cs_index, ...], d, self.ad_inv_o0, self.l0, p_d_g)

        # perturbations in gravity direction, [n_nodes, 3, 3]
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
        :return: Stiffness contribution from gravity forces for lumped masses, [n_lumped, 6, 6]
        """
        # [n_lumped, 3, 3]
        d_g_d_omega = vmap(vec_to_skew, 0, 0)(
            jnp.einsum("ikj,k->ij", rmat[self.m_lumped_index, ...], self.gravity_vec)
        )

        return jnp.zeros_like(self.m_lumped).at[:, :, 3:].set(
            jnp.einsum("ijk,ikl->ijl", self.m_lumped[:, :, 3:], d_g_d_omega))

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
        :param d: Element relative configuration, [n_elem, 6].
        :param p_d: P(d) operator, [n_elem, 6, 12].
        :param eps: Strain vectors, [n_elem, 6].
        :param f_ext_dead: External dead forces in global reference, [n_node, 6].
        :param rmat: Nodal rotation matrices, [n_node, 3, 3].
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12].
        :return: Tangent stiffness matrix with all contributions, [n_dof, n_dof].
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
                k_t_lumped = self._make_k_t_grav_lumped(rmat)
                k_t = self.add_lumped_contributions_to_arr(arr=k_t, lumped_arr=k_t_lumped)
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
            self.m_cs[self.m_cs_index, ...], d, self.ad_inv_o0, self.l0
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
            self.m_cs[self.m_cs_index, ...],
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
        :return: Gyroscopic L and T matrix entries from lumped masses, [n_lumped, 6, 6], [n_lumped, 6, 6]
        """
        c_t_l = vmap(_make_c_t_lumped, (0, 0), 0)(self.m_lumped, v[self.m_lumped_index, ...])  # [n_lumped, 2, 6, 6]
        return c_t_l[:, 0, :, :], c_t_l[:, 1, :, :]

    def _make_sys_matrix(
            self,
            m_t: Array,
            c_t: Array,
            c_t_lumped: Optional[Array],
            k_t: Array,
            t_n: Array,
            ti: TimeIntegrator,
    ) -> Array:
        r"""
        Create the system matrix for the static or dynamic analysis.
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12].
        :param c_t: Disassembled system gyroscopic matrix, [n_elem, 12, 12].
        :param c_t_lumped: Disassembled system lumped gyroscopic matrix, [n_lumped, 6, 6].
        :param k_t: System stiffness matrix, [n_dof, n_dof].
        :param t_n: Tangent operator T(varphi), [n_nodes, 6, 6].
        :param ti: Time integration parameters.
        :return: System matrix, [n_dof, n_dof].
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
            mat = self.add_lumped_contributions_to_arr(arr=mat, lumped_arr=self.m_lumped * ti.beta_prime)
            mat = self.add_lumped_contributions_to_arr(arr=mat, lumped_arr=c_t_lumped * ti.gamma_prime)

        return mat

    def make_f_elem(self, eps: Array) -> Array:
        r"""
        Compute the forces within the elements as :math:`\mathbf{f}_{elem} = \mathcal{K}_{cs} \epsilon`
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Element forces, [n_elem, 6]
        """
        return jnp.einsum('ijk,ik->ij', self.k_cs[self.k_cs_index, ...], eps)

    def make_f_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors
        :param p_d: P(d) operator, [n_elem, 6, 12]
        :param eps: Element strain vectors, [n_elem, 6]
        :return: Internal forces, [n_elem, 12]
        """

        return -jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs[self.k_cs_index, ...], eps)

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
        :return: Lumped gravity force vector, [n_lumped, 6]
        """
        f_rot = jnp.einsum("ikj,k->ij", rmat[self.m_lumped_index, ...], self.gravity_vec)  # [n_lumped, 3]
        f_rot_tot = jnp.concatenate(
            (f_rot, jnp.zeros_like(f_rot)), axis=-1
        )  # [n_lumped, 6]
        return jnp.einsum("ijk,ik->ij", self.m_lumped, f_rot_tot)  # [n_lumped, 6]

    @staticmethod
    def make_f_dead_ext(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, [n_node, 6]
        :param rmat: Deformation rotation matrices, [n_node, 3, 3]
        :return: External forces, [n_node, 6]
        """

        return transform_nodal_vect(f_ext, jnp.swapaxes(rmat, -1, -2))

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
        :param v: Nodal velocities in local frame, [n_node, 6]
        :param v_dot: Nodal accelerations in local frame, [n_node, 6]
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
        :param c_l_lumped: Gyroscopic matrix from lumped masses, [n_lumped, 6, 6]
        :param v: Nodal velocities in local frame, [n_node, 6]
        :param v_dot: Nodal accelerations in local frame, [n_node, 6]
        :return: Inertial forces from lumped masses, [n_lumped, 6]
        """
        f_iner = -jnp.einsum("ijk,ik->ij", self.m_lumped, v_dot[self.m_lumped_index, ...])  # [n_lumped, 6]
        f_gyr = -jnp.einsum("ijk,ik->ij", c_l_lumped, v[self.m_lumped_index, ...])  # [n_lumped, 6]
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
        :param v: Nodal velocities in local frame, [n_node, 6]
        :return: Element relative velocity vectors, [n_elem, 6]
        """

        v_elem = jnp.concatenate(
            (v[self.connectivity[:, 0], :], v[self.connectivity[:, 1], :]), axis=-1
        )  # [n_elem, 12]

        return jnp.einsum("ijk,ik->ij", p_d, v_elem)  # [n_elem, 6]

    @staticmethod
    def make_hg_dot(hg: Array, v: Array) -> Array:
        r"""
        Obtain the time derivative of the nodal coordinates.
        :param hg: Node coordinates, [n_node, 4, 4].
        :param v: Node local velocities, [n_node, 6]
        :return: Coordinate time derivative, [n_node, 4, 4]
        """
        return jnp.einsum('ijk,ikl->ijl', hg, vmap(ha_to_ha_tilde, 0, 0)(v))  # [n_nodes, 4, 4]

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
            approx_gradients: bool = False,
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
            approx_gradients: bool = False,
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
            approx_gradients: bool = False,
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
        Obtain all components of the force from a final solution.
        :param hg: Nodal homogeneous transformation matrices, [n_nodes, 4, 4].
        :param dynamic: Whether to compute dynamic forces.
        :param f_ext_follower: External follower forces in local reference, [n_node, 6].
        :param f_ext_dead: External dead forces in global reference, [n_node, 6].
        :param f_ext_aero: External aero forces in global reference, [n_node, 6].
        :param v: Nodal velocities in global frame, [n_node, 6].
        :param v_dot: Nodal accelerations in global frame, [n_node, 6].
        :param approx_gradients: Whether to stop computing gradients of the inertial and gyroscopic forces with respect to
        the node coordinates, as these are small but nonzero values in practice.
        :return: Configuration vectors, strain vectors, Dead external forces, aero external forces, gravitational forces, internal forces,
        gyroscopic forces, inertial forces and residual forces.
        """

        def prop_grad(x: Array) -> Array:
            return jax.lax.stop_gradient(x) if approx_gradients else x

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
                f_grav_lumped = self._make_f_grav_lumped(hg[:, :3, :3])
                this_f_grav = self.add_lumped_contributions_to_vec(vec=this_f_grav, lumped_vec=f_grav_lumped)
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
                f_iner_lumped, f_gyr_lumped = self._make_f_iner_gyr_lumped(c_l_lumped, v, v_dot)  # type: ignore
                this_f_iner = self.add_lumped_contributions_to_vec(
                    this_f_iner.ravel(), (f_iner_lumped + f_gyr_lumped).ravel()
                ).reshape(-1, 6)
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
        :param solve_dofs: Optional array of degrees of freedom to solve for [n_solve_dofs].
        :param p_d: P(d) operator, [n_elem, 6, 12].
        :param eps: Element strain vectors, [n_elem, 6].
        :param hg: Nodal homogeneous transformation matrices, [n_nodes, 4, 4].
        :param f_ext_follower_n: Nodal follower forces, [n_nodes, 6].
        :param f_ext_dead_n: Nodal dead forces, [n_nodes, 6].
        :param dynamic: Flag for whether to compute dynamic entries.
        :param m_t: Disassembled system mass matrix, [n_elem, 12, 12].
        :param c_l: Dissembled system gyroscopic matrix, [n_elem, 12, 12].
        :param c_l_lumped: Lumped gyroscopic matrix, [n_nodes, 6, 6].
        :param v: Nodal velocities, [n_nodes, 6].
        :param v_dot: Nodal accelerations, [n_node, 6].
        :return: Residual force vector, [n_dof], absolute sum of forces, [n_dof].
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
                f_iner_lumped, f_gyr_lumped = self._make_f_iner_gyr_lumped(c_l_lumped, v, v_dot)
                f_iner_gyr_lumped = (f_iner_lumped + f_gyr_lumped).ravel()
                f_res_vect = self.add_lumped_contributions_to_vec(f_res_vect, f_iner_gyr_lumped)
                f_abs_sum_vect = self.add_lumped_contributions_to_vec(f_abs_sum_vect, jnp.abs(f_iner_gyr_lumped))
            if self.use_gravity:
                f_grav_lumped = self._make_f_grav_lumped(hg[:, :3, :3]).ravel()
                f_res_vect = self.add_lumped_contributions_to_vec(vec=f_res_vect, lumped_vec=f_grav_lumped)
                f_abs_sum_vect = self.add_lumped_contributions_to_vec(vec=f_abs_sum_vect, lumped_vec=f_grav_lumped)

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
        :param hg: Existing nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        :param phi: Perturbation to the configuration vector, [n_nodes, 6]
        :return: Updated nodal homogeneous transformation matrices, [n_nodes, 4, 4]
        """
        return jnp.einsum(
            "ijk,ikl->ijl",
            hg,
            vmap(exp_se3, 0, 0)(phi.reshape(-1, 6)),
        )

    def minimal_states_to_full_states(self, q: StructureMinimalStates) -> StructureFullStates:
        r"""
        Convert a minimal set of states to a useful set of full states.
        :param q: Minimal set of structural states
        :return: Full set of structural states
        """
        hg = self.calculate_hg_from_varphi(q.varphi)
        d = self.make_d(hg=hg)
        eps = self.make_eps(d=d)
        f_elem = self.make_f_elem(eps=eps)
        return StructureFullStates(
            v=q.v, v_dot=q.v_dot, eps=eps, hg=hg, f_elem=f_elem
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
            prescribed_dofs: Sequence[int] | Array | slice | int | None,
            load_steps: int = 1,
            struct_relaxation_factor: float = 1.0,
    ) -> StaticStructure:
        r"""
        Perform static solve of the structure under external loads.
        :param f_ext_follower: External forces array of follower forces [n_node, 6].
        :param f_ext_dead: External forces array of dead loads [n_node, 6].
        :param f_ext_aero: External forces array of aerodynamic loads [n_node, 6].
        :param prescribed_dofs: Index of degrees of freedom which are prescribed (not solved for).
        :param load_steps: Number of load steps to apply the external loads over.
        :param struct_relaxation_factor: Relaxation factor for updates, in range (0, 1].
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

        if not (0.0 < struct_relaxation_factor <= 1.0):
            raise ValueError("struct_relaxation_factor must be in the range (0, 1]")

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=prescribed_dofs_arr)

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
                f_ext_dead_steps, f_ext_aero_steps, i_load_step
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
                    jnp.linalg.solve(k_t_solve_n, f_res_solve_n) * struct_relaxation_factor
            )

            # update configuration, [n_nodes, 4, 4]
            hg_np1_full = self.update_hg(
                hg_n, jnp.zeros(self.n_dof).at[solve_dofs].set(d_varphi_np1)
            )

            # algebra between undeformed and deformed arr_list_shapes, used to check relative convergence, [n_solve_dofs]
            # this is relatively expensive to compute
            if self.struct_convergence_settings.rel_disp_tol is not None:
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

            if VERBOSITY_LEVEL.value >= VerbosityLevel.VERBOSE.value:
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
                        self.struct_convergence_settings,
                    ),
                    hg_init,
                ),
            )

            if VERBOSITY_LEVEL.value >= VerbosityLevel.NORMAL.value:
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
        f_elem = self.make_f_elem(eps=eps)  # compute loads in each element

        return StaticStructure(
            hg=hg,
            conn=self.connectivity,
            o0=self.o0,
            d=d,
            eps=eps,
            varphi=varphi,
            f_int=f_int,
            f_elem=f_elem,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead_local,
            f_ext_aero=f_ext_aero_local,
            f_grav=f_grav,
            f_res=f_res,
            prescribed_dofs=prescribed_dofs_arr,
        )

    @overload
    def base_dynamic_solve(self,
                           struct_case: DynamicStructure,
                           struct_convergence_status: ConvergenceStatus,
                           t: Array,
                           struct_relaxation_factor: float,
                           solve_dofs: Array,
                           load_steps: int,
                           f_ext_dead: Optional[Array],
                           f_ext_follower: Optional[Array],
                           aero_obj: None,
                           aero_case: None,
                           fsi_convergence_status: None,
                           free_wake: None,
                           include_unsteady_aero_force: None,
                           gamma_dot_relaxation_factor: None) -> DynamicStructure:
        ...

    @overload
    def base_dynamic_solve(self,
                           struct_case: DynamicStructure,
                           struct_convergence_status: ConvergenceStatus,
                           t: Array,
                           struct_relaxation_factor: float,
                           solve_dofs: Array,
                           load_steps: int,
                           f_ext_dead: Optional[Array],
                           f_ext_follower: Optional[Array],
                           aero_obj: UVLM,
                           aero_case: DynamicAeroCase,
                           fsi_convergence_status: ConvergenceStatus,
                           free_wake: bool,
                           include_unsteady_aero_force: bool,
                           gamma_dot_relaxation_factor: float) -> DynamicAeroelastic:
        ...

    def base_dynamic_solve(self,
                           struct_case: DynamicStructure,
                           struct_convergence_status: ConvergenceStatus,
                           t: Array,
                           struct_relaxation_factor: float,
                           solve_dofs: Array,
                           load_steps: int,
                           f_ext_dead: Optional[Array],
                           f_ext_follower: Optional[Array],
                           aero_obj: Optional[UVLM],
                           aero_case: Optional[DynamicAeroCase],
                           fsi_convergence_status: Optional[ConvergenceStatus],
                           free_wake: Optional[bool],
                           include_unsteady_aero_force: Optional[bool],
                           gamma_dot_relaxation_factor: Optional[float]) -> DynamicStructure | DynamicAeroelastic:
        r"""
        Generic dynamic solver. Both the structural dynamic solve, and aeroelastic dynamic solve, are formed as wrappers
        of this
        :return:
        """

        if not (0.0 < struct_relaxation_factor <= 1.0):
            raise ValueError("Relaxation factor must be in range (0, 1]")

        n_tstep = len(t)

        include_aero: bool = aero_obj is not None

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

        def _update(
                i_load_step: int,
                i_ts: int,
                struct_converge_status_: ConvergenceStatus,
                hg_n: Array,
                phi_alpha: Array,
                q_alpha: StructureMinimalStates,
                f_ext_aero_alpha_steps: Optional[Array],
        ) -> tuple[
            int,
            int,
            ConvergenceStatus,
            Array,
            Array,
            StructureMinimalStates,
            Optional[Array],
        ]:
            r"""
            Solution update for a single iteration of the nonlinear solver at a given time step and load step.
            :param i_load_step: Load step index.
            :param i_ts: Time step index.
            :param struct_converge_status_: ConvergenceStatus object for the current iteration, used to track
            convergence and print messages.
            :param hg_n: Transformation matrices at iteration varphi, [n_nodes, 4, 4].
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
                f_ext_dead_alpha_steps[:, i_ts, :, :] if f_ext_dead_alpha_steps is not None else None,
                f_ext_aero_alpha_steps, i_load_step
            )  # [n_node, 6]

            k_t = self._make_k_t_full(
                d_n,
                p_d_n,
                eps_n,
                total_f_ext_dead,
                hg_update[:, :3, :3],
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
                solve_dofs=solve_dofs,
                p_d=p_d_n,
                eps=eps_n,
                hg=hg_update,
                f_ext_follower_n=f_ext_follower_alpha_steps[i_load_step, i_ts, ...]
                if f_ext_follower_alpha_steps is not None
                else None,
                f_ext_dead_n=total_f_ext_dead,
                dynamic=True,
                m_t=m_t,
                c_l=c_l,
                c_l_lumped=c_l_lumped,
                v=q_alpha.v,
                v_dot=q_alpha.v_dot,
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
            d_n_np1 = jnp.linalg.solve(sys_mat, f_res_n_solve) * struct_relaxation_factor
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
            struct_converge_status_.update(
                delta_disp=d_n_np1,
                total_disp=phi_np1,
                delta_force=f_res_n_solve,
                total_force=f_abs_sum_n,
            )

            if VERBOSITY_LEVEL.value >= VerbosityLevel.VERBOSE.value:
                struct_converge_status_.print_struct_message(t[i_ts], i_load_step)

            q_alpha_update = StructureMinimalStates(
                varphi=None, v=v_np1, v_dot=v_dot_np1, a=q_alpha.a
            )

            return i_load_step, i_ts, struct_converge_status_, hg_n, phi_np1, q_alpha_update, f_ext_aero_alpha_steps

        @overload
        def time_step_loop(
                i_ts: int,
                struct_sol: DynamicStructure,
                struct_converge_status: ConvergenceStatus,
                aero_sol: None,
                fsi_convergence_status_: None,
        ) -> tuple[DynamicStructure, ConvergenceStatus, None, None]:
            ...

        @overload
        def time_step_loop(
                i_ts: int,
                struct_sol: DynamicStructure,
                struct_converge_status: ConvergenceStatus,
                aero_sol: DynamicAeroCase,
                fsi_convergence_status_: ConvergenceStatus,
        ) -> tuple[DynamicStructure, ConvergenceStatus, DynamicAeroCase, ConvergenceStatus]:
            ...

        def time_step_loop(
                i_ts: int,
                struct_sol: DynamicStructure,
                struct_converge_status: ConvergenceStatus,
                aero_sol: Optional[DynamicAeroCase],
                fsi_convergence_status_: Optional[ConvergenceStatus],
        ) -> tuple[DynamicStructure, ConvergenceStatus, Optional[DynamicAeroCase], Optional[ConvergenceStatus]]:
            r"""
            Performs analysis on a single time step, including load stepping
            :param i_ts: Index of time step to solve
            :param struct_sol: Solution object, with results up to time step i_ts-1.
            :param struct_converge_status: Convergence status object.
            :param aero_sol: Aero solution object, with results up to time step i_ts-1, if aero is included.
            :param fsi_convergence_status_: Convergence status object.
            :return: Solution object with results up to time step i_ts.
            """

            # predictor step
            phi_init, q_init = self.time_integrator.predict_q(struct_sol.get_minimal_states(i_ts - 1))
            phi_alpha_init, q_alpha_init = self.time_integrator.calculate_q_alpha(
                q_nm1=struct_sol.get_minimal_states(i_ts - 1), q_n=q_init, phi_n=phi_init
            )

            q_alpha_init.varphi = None  # this value is not used during the loop

            if include_aero:
                if aero_sol is None or fsi_convergence_status_ is None or struct_sol.f_ext_aero is None:
                    raise ValueError("Missing aero arguments")

                fsi_convergence_status_.reset_status()

                # f_ext_aero is stored in local frame, so we convert back to global
                # so that both operands of the alpha blend are in the same (global) frame.
                f_aero_nm1 = jnp.concatenate([
                    jnp.einsum("ijk,ik->ij", struct_sol.hg[i_ts - 1, :, :3, :3],
                               struct_sol.f_ext_aero[i_ts - 1, :, :3]),
                    jnp.einsum("ijk,ik->ij", struct_sol.hg[i_ts - 1, :, :3, :3],
                               struct_sol.f_ext_aero[i_ts - 1, :, 3:]),
                ], axis=-1)

                _, struct_sol, aero_sol, struct_converge_status, fsi_convergence_status_, phi_alpha, q_alpha, _, _ = jax.lax.while_loop(
                    lambda args_: ~cast(ConvergenceStatus, args_[4]).get_status(),
                    lambda args_: fsi_convergence_loop(*args_),
                    (i_ts,
                     struct_sol,
                     aero_sol,
                     struct_converge_status,
                     fsi_convergence_status_,
                     phi_alpha_init,
                     q_alpha_init,
                     f_aero_nm1,  # this value is for the previous timesteps force, and is propagated unaltered
                     f_aero_nm1,  # first guess for forcing at alpha is to use value from i_ts=n-1
                     ))

            else:
                # solve pure structural problem
                _, struct_converge_status, hg, phi_alpha, q_alpha, _ = load_step_loop(
                    i_ts=i_ts,
                    struct_converge_status=struct_converge_status,
                    hg_alpha=struct_sol.hg[i_ts - 1, ...],
                    phi_alpha=phi_alpha_init,
                    q_alpha=q_alpha_init,
                    f_ext_aero_steps=None
                )

            # print message where we only require one message per timestep
            if VERBOSITY_LEVEL.value == VerbosityLevel.NORMAL.value:
                struct_converge_status.print_struct_message(t=struct_sol.t[i_ts], i_load_step=load_steps - 1)
                if include_aero and fsi_convergence_status_ is not None:
                    fsi_convergence_status_.print_fsi_message(t=struct_sol.t[i_ts])

            # postprocess results for time step and store in solution object
            q_n, phi_n = self.time_integrator.calculate_q_n_from_q_alpha(
                q_alpha=q_alpha,
                q_nm1=struct_sol.get_minimal_states(i_ts - 1),
                phi_alpha=phi_alpha,
            )

            # update pseudo-acceleration
            q_n.a = self.time_integrator.calculate_a_n(
                a_nm1=struct_sol.a[i_ts - 1, ...],
                v_dot_nm1=struct_sol.v_dot[i_ts - 1, ...],
                v_dot_n=q_n.v_dot,
            )

            # final node coordinates
            hg_n = self.update_hg(struct_sol.hg[i_ts - 1, ...], phi_n)

            if include_aero:
                if aero_sol is None or aero_obj is None or fsi_convergence_status_ is None or free_wake is None or include_unsteady_aero_force is None:
                    raise ValueError("Missing aero arguments")

                f_ext_aero = aero_sol.project_forcing_to_beam(i_ts=i_ts, rmat=hg_n[:, :3, :3], x0_aero=aero_obj.x0_b,
                                                              include_unsteady=include_unsteady_aero_force)


            else:
                f_ext_aero = None

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
                f_ext_aero=f_ext_aero,
                v=q_n.v,
                v_dot=q_n.v_dot,
            )
            struct_sol.d = struct_sol.d.at[i_ts, ...].set(d)
            struct_sol.eps = struct_sol.eps.at[i_ts, ...].set(eps)
            struct_sol.v = struct_sol.v.at[i_ts, ...].set(q_n.v)
            struct_sol.v_dot = struct_sol.v_dot.at[i_ts, ...].set(q_n.v_dot)
            struct_sol.a = struct_sol.a.at[i_ts, ...].set(q_n.a)
            struct_sol.hg = struct_sol.hg.at[i_ts, ...].set(hg_n)
            struct_sol.varphi = struct_sol.varphi.at[i_ts, ...].set(vmap(hg_to_d, (0, 0), 0)(self.hg0, hg_n))

            if f_ext_follower is not None:
                struct_sol.f_ext_follower = struct_sol.f_ext_follower.at[i_ts, ...].set(
                    f_ext_follower[i_ts, ...]
                )
            if f_ext_dead is not None:
                struct_sol.f_ext_dead = struct_sol.f_ext_dead.at[i_ts, ...].set(f_ext_dead_local)

            if f_ext_aero is not None:
                struct_sol.f_ext_aero = struct_sol.f_ext_aero.at[i_ts, ...].set(f_ext_aero_local)

            if self.use_gravity:
                if struct_sol.f_grav is None: raise ValueError("struct_sol.f_grav is None")
                struct_sol.f_grav = struct_sol.f_grav.at[i_ts, ...].set(f_grav)
            struct_sol.f_int = struct_sol.f_int.at[i_ts, ...].set(f_int)
            struct_sol.f_elem = struct_sol.f_elem.at[i_ts, ...].set(self.make_f_elem(eps=eps))
            struct_sol.f_iner_gyr = struct_sol.f_iner_gyr.at[i_ts, ...].set(f_iner + f_gyr)
            struct_sol.f_res = struct_sol.f_res.at[i_ts, ...].set(f_res)

            return struct_sol, struct_converge_status, aero_sol, fsi_convergence_status_

        def fsi_convergence_loop(i_ts: int,
                                 struct_sol: DynamicStructure,
                                 aero_sol: DynamicAeroCase,
                                 struct_converge_status: ConvergenceStatus,
                                 fsi_converge_status: ConvergenceStatus,
                                 phi_alpha_init: Array,
                                 q_alpha_init: StructureMinimalStates,
                                 f_aero_nm1: Array,
                                 f_aero_alpha_prev: Array) -> tuple[
            int, DynamicStructure, DynamicAeroCase, ConvergenceStatus, ConvergenceStatus, Array, StructureMinimalStates, Array, Array]:

            # obtain coordinates at timestep (not alpha)
            phi_n = self.time_integrator.calculate_phi_from_phi_alpha(phi_alpha=phi_alpha_init)
            v_n = self.time_integrator.calculate_v_from_v_alpha(v_alpha=q_alpha_init.v,
                                                                v_nm1=struct_sol.v[i_ts - 1, ...])

            hg_n = self.update_hg(hg=struct_sol.hg[i_ts - 1, ...], phi=phi_n)
            hg_dot = self.make_hg_dot(hg=hg_n, v=v_n)

            if (aero_obj is None or free_wake is None or struct_sol.f_ext_aero is None
                    or include_unsteady_aero_force is None or gamma_dot_relaxation_factor is None):
                raise ValueError("Missing aero parameters")

            # evaluate aerodynamic forcing on beam
            aero_sol = aero_obj.case_solve(case=aero_sol, i_ts=i_ts, hg=hg_n, hg_dot=hg_dot, static=False,
                                           free_wake=free_wake, horseshoe=False,
                                           gamma_dot_relaxation=gamma_dot_relaxation_factor)

            f_aero_n = aero_sol.project_forcing_to_beam(i_ts=i_ts, rmat=hg_n[:, :3, :3], x0_aero=aero_obj.x0_b,
                                                        include_unsteady=include_unsteady_aero_force)

            # aerodynamic force at alpha point, subsequently divided into load steps
            f_aero_alpha = self.time_integrator.calculate_f_alpha(f_nm1=f_aero_nm1, f_n=f_aero_n)

            f_aero_alpha_steps = self._make_load_steps_f(f=f_aero_alpha, weighting=load_step_weight,
                                                         apply_alpha_weighting=False)

            # reset convergence status
            struct_converge_status.reset_status()

            # solve structural problem for given aero load
            _, struct_converge_status, hg_out, phi_alpha, q_alpha, _ = load_step_loop(
                i_ts,
                struct_converge_status,
                struct_sol.hg[i_ts - 1, ...],
                phi_alpha_init,
                q_alpha_init,
                f_aero_alpha_steps
            )

            # update the FSI convergence object
            # note that for convenience we use the alpha properties
            fsi_converge_status.update(
                delta_disp=(phi_alpha_init - phi_alpha).ravel()[solve_dofs],
                total_disp=phi_alpha.ravel()[solve_dofs],
                delta_force=(f_aero_alpha - f_aero_alpha_prev).ravel()[solve_dofs],
                total_force=f_aero_alpha.ravel()[solve_dofs],
            )

            if VERBOSITY_LEVEL.value >= VerbosityLevel.VERBOSE.value:
                fsi_converge_status.print_fsi_message(t=t[i_ts])

            return i_ts, struct_sol, aero_sol, struct_converge_status, fsi_converge_status, phi_alpha, q_alpha, f_aero_nm1, f_aero_alpha

        def struct_convergence_loop(
                i_load_step: int,
                i_ts: int,
                struct_converge_status: ConvergenceStatus,
                hg_alpha: Array,
                phi_alpha: Array,
                q_alpha: StructureMinimalStates,
                f_ext_aero_steps: Optional[Array]
        ) -> tuple[int, ConvergenceStatus, Array, Array, StructureMinimalStates, Optional[Array]]:
            r"""
            Convergence loop within each load step of a time step.
            :param i_load_step: Load step index.
            :param i_ts: Time step index.
            :param struct_converge_status: ConvergenceStatus object to update with convergence information during load
            stepping.
            :param hg_alpha: Node transformations at the beginning of the load step, [n_nodes, 4, 4].
            :param phi_alpha: Node configuration increments in algebra space, [n_nodes, 6].
            :param q_alpha: Minimal states at intermediate alpha step.
            :param f_ext_aero_steps: Optional aerodynamic forcing alpha load steps [n_steps, n_nodes, 6].
            :return: Time step index, convergence status, and updated configuration, velocities, accelerations, and
            optional aerodynamic forcing.
            """

            struct_converge_status.reset_status()

            _, _, struct_converge_status, hg_solve, phi_alpha, q_alpha, _ = jax.lax.while_loop(
                lambda args_: ~args_[2].get_status(),
                lambda args_: _update(*args_),
                (i_load_step, i_ts, struct_converge_status, hg_alpha, phi_alpha, q_alpha, f_ext_aero_steps),
            )

            if VERBOSITY_LEVEL.value >= VerbosityLevel.VERBOSE.value:
                struct_converge_status.print_struct_message(t[i_ts], i_load_step)

            return i_ts, struct_converge_status, hg_solve, phi_alpha, q_alpha, f_ext_aero_steps

        def load_step_loop(
                i_ts: int,
                struct_converge_status: ConvergenceStatus,
                hg_alpha: Array,
                phi_alpha: Array,
                q_alpha: StructureMinimalStates,
                f_ext_aero_steps: Optional[Array]
        ) -> tuple[int, ConvergenceStatus, Array, Array, StructureMinimalStates, Optional[Array]]:
            r"""
            Performs load stepping iterations for a given time step
            :param i_ts: Timestep index for which to perform load stepping
            :param struct_converge_status: ConvergenceStatus object to update with load stepping convergence information
            :param hg_alpha: SE(3) nodal transformation matrices at the beginning of the load step, [n_nodes, 4, 4]
            :param phi_alpha: Nodal updates to the configuration in the algebra space, [n_nodes, 6]
            :param q_alpha: Minimal states at intermediate alpha step
            :param f_ext_aero_steps: Optional aerodynamic forcing alpha load steps [n_steps, n_nodes, 6]
            :return: Time step index, updated ConvergenceStatus object, and updated configuration, velocities and accelerations after load stepping
            """
            return jax.lax.fori_loop(
                0,
                load_steps,
                lambda i_load_step, args: struct_convergence_loop(i_load_step, *args),
                (i_ts, struct_converge_status, hg_alpha, phi_alpha, q_alpha, f_ext_aero_steps),
            )

        struct_case, _, aero_case, _ = jax.lax.fori_loop(
            1,
            n_tstep,
            lambda i_ts, args: time_step_loop(i_ts, *args),
            (struct_case, struct_convergence_status, aero_case, fsi_convergence_status),
        )

        if include_aero:
            if aero_case is None: raise ValueError("aero_case cannot be None")

            from aegrad.coupled.data_structures import DynamicAeroelastic  # import here to prevent circular references
            return DynamicAeroelastic(structure=struct_case, aero=aero_case)
        else:
            return struct_case

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
            struct_relaxation_factor: float = 1.0,
            spectral_radius: float = 0.9,
    ) -> DynamicStructure:
        r"""
        Perform dynamic solve of the structure under external loads
        :param init_state: Initial state of the structure, either as a DynamicStructureSnapshot or StaticStructure. If
        None, the reference configuration is used with zero velocities.
        :param n_tstep: Number of time steps to simulate.
        :param dt: Time step length.
        :param f_ext_follower: Following external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external follower forces.
        :param f_ext_dead: Dead external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external dead forces.
        :param f_ext_aero: Aerodynamic external forces array, [n_tstep, n_node, 6], [n_node, 6] or None for zero external aerodynamic forces.
        :param prescribed_dofs: Degrees of freedom which are prescribed (not solved for).
        :param load_steps: Number of load steps to apply the external loads over.
        :param struct_relaxation_factor: Relaxation factor for Newton-Raphson iterations, in the range (0, 1].
        :param spectral_radius: Spectral radius for the time integrator, in the range [0, 1].
        :return: DynamicStructure dataclass containing results of the dynamic analysis.
        """

        # add a warning if using 32-bit floats
        warn_if_32_bit()

        if load_steps <= 0:
            raise ValueError("load_steps must be a positive integer")

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs = get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=prescribed_dofs_arr)

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
        self.time_integrator = TimeIntegrator(spectral_radius=spectral_radius, dt=dt)

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

            f_elem = self.make_f_elem(eps=eps)

            return DynamicStructureSnapshot(
                hg=init_state__.hg,
                conn=self.connectivity,
                o0=self.o0,
                d=d,
                eps=eps,
                varphi=init_state__.varphi,
                v=init_state__.v,
                v_dot=init_state__.v_dot,
                a=init_state__.v_dot,  # initial pseudo-acceleration set equal to initial acceleration
                f_ext_follower=init_state__.f_ext_follower,
                f_ext_dead=f_ext_dead_,
                f_ext_aero=f_ext_aero_,
                f_grav=f_grav,
                f_int=f_int,
                f_elem=f_elem,
                f_iner_gyr=f_iner + f_gyr,  # type: ignore
                f_res=f_res,
                t=init_state__.t,
                i_ts=init_state__.i_ts,
                prescribed_dofs=prescribed_dofs_arr,
            )

        # time steps
        t = jnp.arange(n_tstep) * dt
        if isinstance(init_state, DynamicStructure):
            t += init_state.t[0]
        elif isinstance(init_state, DynamicStructureSnapshot):
            t += init_state.t

        # set up initial state
        if init_state is None:
            init_state_: DynamicStructureSnapshot = self.reference_configuration(
                use_f_aero=f_ext_aero is not None,
                use_f_ext_dead=f_ext_dead is not None,
                use_f_ext_follower=f_ext_follower is not None,
            ).to_dynamic(t=None)
        elif isinstance(init_state, StaticStructure):
            init_state_ = init_state.to_dynamic(t=None)
        elif isinstance(init_state, DynamicStructureSnapshot):
            init_state_ = init_state
        else:
            raise TypeError("Invalid init_state type")

        # check if initial state satisfies equilibrium
        init_state_eval = evaluate_initial_equilibrium(init_state_)
        dynamic_struct = DynamicStructure.initialise(
            initial_snapshot=init_state_eval, t=t, use_f_ext_follower=f_ext_follower is not None,
            use_f_ext_dead=f_ext_dead is not None, use_f_ext_aero=False
        )
        converge_status = ConvergenceStatus(
            convergence_settings=self.struct_convergence_settings
        )

        return self.base_dynamic_solve(dynamic_struct,
                                       converge_status,
                                       t,
                                       struct_relaxation_factor,
                                       solve_dofs,
                                       load_steps,
                                       f_ext_dead,
                                       f_ext_follower,
                                       None,
                                       None,
                                       None,
                                       None, None, None)
