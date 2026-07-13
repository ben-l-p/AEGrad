from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Optional

from jax import Array, vmap
from jax import numpy as jnp

from jax.scipy.linalg import block_diag

from aegrad.algebra.array_utils import check_arr_shape
from aegrad.algebra.se3 import exp_se3
from aegrad.structure.linear.data_structures import (
    BeamLinearResult,
    BeamInputUnflattened,
    BeamStateUnflattened,
    BeamOutputUnflattened,
)
from aegrad.structure.utils import transform_nodal_vect
from aegrad.utils.constants import BASE_LOBATTO_ORDER
from aegrad.structure.data_structures import StaticStructure
from aegrad.utils.linear import LinearModel, LinearComponent, SliceEntry, LinearSystem

if TYPE_CHECKING:
    from aegrad.structure.beam import BaseBeamStructure


class LinearBeam(LinearModel):
    r"""
    Class to represent a linearised beam system about a reference state.
    """

    def __init__(
        self,
        beam: BaseBeamStructure,
        reference: StaticStructure,
        dt: float | Array,
        local_forcing: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
    ):
        self.free_dofs: tuple = reference.free_dofs
        self.n_free_dof: int = len(reference.free_dofs)
        self.n_nodes: int = beam.n_nodes

        self.local_forcing: bool = local_forcing

        super().__init__(reference=reference, dt=dt)

        self.m_global, self.k_global = beam.make_global_m_k(
            case=reference, int_order=int_order, local_forcing=local_forcing
        )

        self.sys: LinearSystem = self.linearise()

    @property
    def reference(self) -> StaticStructure:
        assert isinstance(self._reference, StaticStructure), (
            "Reference state must be of type StaticStructure."
        )
        return self._reference

    def extract_reference_inputs(
        self,
    ) -> dict[str, Optional[Array]]:
        return {
            "f_ext_follower": self.reference.f_ext_follower,
            "f_ext_dead": self.reference.f_ext_dead,
        }

    def extract_reference_states(
        self,
    ) -> dict[str, Optional[Array]]:

        return {
            "q": jnp.zeros(len(self.reference.free_dofs)),
            "q_dot": jnp.zeros(len(self.reference.free_dofs)),
        }

    def extract_reference_outputs(
        self,
    ) -> dict[str, Optional[Array]]:
        return {
            "q": jnp.zeros(len(self.reference.free_dofs)),
            "q_dot": jnp.zeros(len(self.reference.free_dofs)),
        }

    @property
    def input_object(self) -> type[BeamInputUnflattened]:
        return BeamInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[BeamStateUnflattened]:
        return BeamStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[BeamOutputUnflattened]:
        return BeamOutputUnflattened

    def _make_input_slices(
        self,
        reference: StaticStructure,
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            SliceEntry(
                name="f_ext_follower",
                enabled=True,
                shapes=(self.n_nodes, 6),
            ),
            SliceEntry(
                name="f_ext_dead",
                enabled=True,
                shapes=(self.n_nodes, 6),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: StaticStructure
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry(name="q", enabled=True, shapes=(self.n_free_dof,)),
            SliceEntry(name="q_dot", enabled=True, shapes=(self.n_free_dof,)),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: StaticStructure
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry(name="q", enabled=True, shapes=(self.n_free_dof,)),
            SliceEntry(name="q_dot", enabled=True, shapes=(self.n_free_dof,)),
        )
        return self._make_slices(slice_entries)

    def linearise(self) -> LinearSystem:
        cont_sys = self.linearise_continuous()
        return cont_sys.continuous_to_discrete(method="tustin")

    def linearise_continuous(self) -> LinearSystem:
        r"""
        Form a system of linear equations about a reference state. The system is of the form:
        :math:`\dot{\mathbf{x}} = \mathbf{A~x + B~u}, \mathbf{y} = \mathbf{C~x + D~u}`. Input vector :math:`\mathbf{u}`
        is the packed vector of follower and dead applied force perturbations at nodal degrees of freedom (each
        [n_nodes, 6]). Output vector :math:`\mathbf{y}` is the vector of displacements, and state vector
        :math:`\mathbf{x}` is the vector of displacements and velocities.

        The :math:`\mathbf{B}` operator applies the frame rotation between the user-facing follower/dead force inputs
        and the internal beam frame determined by ``local_forcing``:

        * ``local_forcing=True``: internal frame is local. Follower forces pass through unchanged and dead forces are
          rotated global-to-local via :math:`\mathbf{R}^T`.
        * ``local_forcing=False``: internal frame is global. Dead forces pass through unchanged and follower forces are
          rotated local-to-global via :math:`\mathbf{R}`.

        :return: Linearised system.
        """

        m_inv = jnp.linalg.inv(self.m_global)

        rmat = self.reference.hg[:, :3, :3]  # per-node local-to-global rotation
        rmat_t = jnp.swapaxes(rmat, -1, -2)  # per-node global-to-local rotation

        free_dofs_arr = jnp.array(self.free_dofs)

        def _block_rot(rmat_per_node: Array) -> Array:
            chi = jnp.zeros((self.n_nodes, 6, 6))
            chi = chi.at[:, :3, :3].set(rmat_per_node)
            chi = chi.at[:, 3:, 3:].set(rmat_per_node)
            return block_diag(*chi)

        eye_nodal = jnp.eye(self.n_nodes * 6)
        if self.local_forcing:
            follower_rot = eye_nodal
            dead_rot = _block_rot(rmat_t)
        else:
            follower_rot = _block_rot(rmat)
            dead_rot = eye_nodal

        p_free = eye_nodal[free_dofs_arr, :]  # (n_free_dof, n_nodes * 6)

        zero_ff = jnp.zeros((self.n_free_dof, self.n_free_dof))
        eye_ff = jnp.eye(self.n_free_dof)

        a = jnp.block(
            [
                [zero_ff, eye_ff],
                [-m_inv @ self.k_global, zero_ff],
            ]
        )

        b_bottom = m_inv @ p_free @ jnp.concatenate([follower_rot, dead_rot], axis=1)
        b = jnp.concatenate(
            [jnp.zeros((self.n_free_dof, self.n_inputs)), b_bottom], axis=0
        )

        c = jnp.concatenate([eye_ff, zero_ff], axis=1)
        d = jnp.zeros((self.n_free_dof, self.n_inputs))

        return LinearSystem(
            a=a, b=b, c=c, d=d, dt=self.dt, continuous_time=True, removed_u_np1=False
        )

    def run(
        self,
        u: BeamInputUnflattened,
        x0: Optional[BeamStateUnflattened] = None,
    ) -> BeamLinearResult:
        r"""
        Run the linear system.
        :param u: Total input over time (reference + pertubation).
        :param x0: Initial state perturbations, defaults to zero state.
        :return: Linear system results.
        """

        assert isinstance(self.reference, StaticStructure), (
            "Invalid reference structure"
        )

        # has to be specified in the input object, as the linear system does not know how many time steps to run for in
        # the degenerate case where there is no external forcing
        n_tstep = u.n_tstep

        # initial states
        if x0 is None:
            q0_ = None
            q0_dot_ = None
        else:
            q0_ = x0.q
            q0_dot_ = x0.q_dot

        if q0_ is None:
            q0 = jnp.zeros((self.n_free_dof,))
        else:
            q0 = q0_.ravel()[self.free_dofs]
            check_arr_shape(q0, (self.n_free_dof,), name="q0")

        if q0_dot_ is None:
            q0_dot = jnp.zeros((self.n_free_dof,))
        else:
            q0_dot = q0_dot_.ravel()[self.free_dofs]
            check_arr_shape(q0_dot, (self.n_free_dof,), name="q0_dot")

        def create_delta_forcing(
            reference_f: Optional[Array],
            input_f_t: Optional[Array],
            name: str,
        ) -> Array:
            r"""
            Compute the perturbation forcing time history relative to the reference.
            :param reference_f: Reference forcing vector, [n_nodes, 6] or None.
            :param input_f_t: Input forcing vector, [n_tstep, n_nodes, 6] or None.
            :param name: Name of the forcing vector for error messages.
            :return: Delta forcing vector, [n_tstep, n_nodes, 6].
            """
            if input_f_t is None:
                if reference_f is None:
                    return jnp.zeros((n_tstep, self.n_nodes, 6))
                return -jnp.broadcast_to(
                    reference_f[None, :, :],
                    (n_tstep, self.n_nodes, 6),
                )
            check_arr_shape(input_f_t, (n_tstep, self.n_nodes, 6), name=name)
            if reference_f is None:
                return input_f_t
            return input_f_t - reference_f[None, :, :]

        # reference dead forces are stored in the local frame and so are rotated back
        rmat_ref = self.reference.hg[:, :3, :3]
        reference_dead_global = (
            transform_nodal_vect(self.reference.f_ext_dead, rmat_ref)
            if self.reference.f_ext_dead is not None
            else None
        )

        delta_f_ext_follower_t = create_delta_forcing(
            reference_f=self.reference.f_ext_follower,
            input_f_t=u.f_ext_follower,
            name="f_ext_follower_t",
        )

        delta_f_ext_dead_t = create_delta_forcing(
            reference_f=reference_dead_global,
            input_f_t=u.f_ext_dead,
            name="f_ext_dead_t",
        )

        delta_u = BeamInputUnflattened(
            n_tstep=n_tstep,
            f_ext_follower=delta_f_ext_follower_t,
            f_ext_dead=delta_f_ext_dead_t,
        )
        delta_u_vec = self._pack_input_vector_t(delta_u)

        # run linear system
        x_t, y_t = self.sys.run(
            u=delta_u_vec,
            x0=jnp.concatenate((q0, q0_dot)),
        )

        # extract perturbations in displacements and velocities
        delta_q_t = x_t[:, : self.n_free_dof]
        delta_q_dot_t = x_t[:, self.n_free_dof :]

        # add in zeros for dofs not solved for to allow for reconstructing the full configuration
        delta_q_t_full = (
            jnp.zeros((n_tstep, self.n_nodes * 6))
            .at[:, self.free_dofs]
            .set(delta_q_t)
            .reshape(n_tstep, self.n_nodes, 6)
        )

        delta_hg_t = vmap(vmap(exp_se3, 0, 0), 1, 1)(
            delta_q_t_full
        )  # [n_tstep, n_nodes, 4, 4]

        hg_t = jnp.einsum("ijk,hikl->hijl", self.reference.hg, delta_hg_t)

        return BeamLinearResult(
            reference=self.reference,
            f_ext_follower=u.f_ext_follower,
            f_ext_dead=u.f_ext_dead,
            delta_q=delta_q_t,
            delta_q_dot=delta_q_dot_t,
            hg=hg_t,
            t=jnp.arange(n_tstep) * self.dt,
        )
