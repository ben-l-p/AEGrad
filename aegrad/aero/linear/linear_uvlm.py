from __future__ import annotations
from typing import Sequence, TYPE_CHECKING, Optional
from functools import reduce
from operator import mul
from enum import Enum
from jax import Array, jit, vmap
import jax
import jax.numpy as jnp
import os
from pathlib import Path

from aegrad.aero.data_structures import DynamicAeroCase, AeroSnapshot
from aegrad.aero.linear.data_structures import (
    InputUnflattened,
    StateUnflattened,
    OutputUnflattened,
)
from aegrad.aero.linear.data_structures import (
    _LinearComponent,
    _SliceEntry,
    InputSlices,
    StateSlices,
    OutputSlices,
    AeroLinearResult,
)
from aegrad.aero.utils import (
    compute_c,
    compute_nc,
    propagate_wake,
    calculate_steady_forcing,
)
from aegrad.algebra.linear_operators import LinearOperator, LinearSystem
from aegrad.algebra.array_utils import ArrayList, split_to_vertex
from aegrad.aero.flowfields import FlowField
from aegrad.aero.utils import biot_savart_cutoff, KernelFunction
from aegrad.utils import _shallow_asdict
from aegrad.print_utils import warn
from aero.aic import compute_aic_solve, compute_v_ind

if TYPE_CHECKING:
    from aegrad.aero.uvlm import UVLM


class LinearWakeType(Enum):
    # (is prescribed, is free)
    FROZEN = (False, False)
    PRESCRIBED = (True, False)
    FREE = (True, True)


class LinearUVLM:
    r"""
    Class to represent a linearised UVLM aerodynamic system about a reference state.
    """

    def __init__(
        self,
        case: UVLM,
        reference: AeroSnapshot,
        wake_type: LinearWakeType = LinearWakeType.FREE,
        bound_upwash: bool = True,
        wake_upwash: bool = True,
        unsteady_force: bool = True,
        gamma_dot_state: bool = False,
    ):
        r"""
        Initialize linear UVLM system about a reference state.
        :param case: UVLM case object to linearise.
        :param reference: StaticAero representing the reference state for linearisation.
        :param wake_type: Instance of LinearWakeType enum to specify wake treatment.
        :param bound_upwash: If true, include bound surface upwash velocities as inputs.
        :param wake_upwash: If true, include wake surface upwash velocities as inputs.
        :param unsteady_force: If true, include unsteady force.
        :param gamma_dot_state: If true, include bound circulation time derivative as a state.
        """

        # options
        self.prescribed_wake, self.free_wake = wake_type.value
        self.unsteady_force: bool = unsteady_force
        self.bound_upwash: bool = bound_upwash
        self.wake_upwash: bool = wake_upwash
        self.gamma_dot_state: bool = gamma_dot_state

        # save names from case
        self.surf_b_names: list[str] = [f"linear_{name}" for name in case.surf_b_names]
        self.surf_w_names: list[str] = [f"linear_{name}" for name in case.surf_w_names]

        # mirroring info
        self.mirror_point: Optional[Array] = case.mirror_point
        self.mirror_normal: Optional[Array] = case.mirror_normal

        # time info
        self.dt: Array = case.dt

        # check that the reference state is steady
        # whilst linearisation can be performed about unsteady states, the current implementation omits some terms
        # required for this, however, cannot see a practical use case for such a model. Warn the user if the reference
        # state appears unsteady.
        if max([jnp.abs(zbd).max() for zbd in reference.zeta_b_dot]) > 1e-6:
            warn(
                "Reference bound surface velocities are non-zero. Ensure that the reference state is steady for linearisation."
            )

        if max([jnp.abs(gbd).max() for gbd in reference.gamma_b_dot]) > 1e-6:
            warn(
                "Reference bound circulation time derivative is non-zero. Ensure that the reference state is steady for linearisation."
            )

        # reference state
        self.reference: AeroSnapshot = reference

        # slices of individial surface components in full vector
        self.input_slices, self.n_inputs = self._make_input_slices()
        self.state_slices, self.n_states = self._make_state_slices()
        self.output_slices, self.n_outputs = self._make_output_slices()

        # kernels
        self.kernels_b: Sequence[KernelFunction] = self.reference.n_surf * [
            biot_savart_cutoff
        ]
        self.kernels_w: Sequence[KernelFunction] = self.reference.n_surf * [
            biot_savart_cutoff
        ]

        # wake propagation deltas
        self.delta_w: Sequence[Optional[Array]] = case.delta_w

        # linear operators for system
        self.base_sys: LinearSystem = self.linearise()

        # final system - this is overwritten for updating models
        self.sys: LinearSystem = self.base_sys

    def get_reference_inputs(self) -> InputUnflattened:
        r"""
        Get the reference input state about which the system is linearised.
        :return: InputUnflattened object representing the reference inputs.
        """
        return InputUnflattened(
            self.reference.zeta_b,
            self.reference.zeta_b_dot,
            ArrayList.zeros_like(self.reference.zeta_b) if self.bound_upwash else None,
            ArrayList.zeros_like(self.reference.zeta_w) if self.wake_upwash else None,
        )

    def get_reference_states(self) -> StateUnflattened:
        r"""
        Get the reference state about which the system is linearised.
        :return: StateUnflattened object representing the reference states.
        """
        return StateUnflattened(
            self.reference.gamma_b,
            self.reference.gamma_w,
            self.reference.gamma_b if self.unsteady_force else None,
            self.reference.gamma_b_dot if self.gamma_dot_state else None,
            self.reference.zeta_w if self.prescribed_wake else None,
            self.reference.zeta_b if self.prescribed_wake else None,
        )

    def get_reference_outputs(self) -> OutputUnflattened:
        r"""
        Get the reference outputs about which the system is linearised.
        :return: OutputUnflattened object representing the reference outputs.
        """
        return OutputUnflattened(
            self.reference.f_steady,
            self.reference.f_unsteady if self.unsteady_force else None,
        )

    @staticmethod
    def _make_slices[T](
        slice_entries: Sequence[_SliceEntry], cls: type[T]
    ) -> tuple[T, int]:
        r"""
        Helper function to create slices classes for the vectors, and count the number of elements.
        Blocks should be passed in the int_order they are in the dataclass.
        :param slice_entries: Sequence of _SliceEntry objects defining the slices.
        :param cls: The class type to instantiate for the slices, e.g. InputSlices.
        :return: Tuple of (slices class instance, total number of elements).
        """
        # make slices
        cnt = 0
        out_dict = {}
        for entry in slice_entries:
            if not entry.enabled:  # if disabled
                out_dict[entry.name] = _LinearComponent(False, None, None)
            else:
                slices = []
                for size in [reduce(mul, shape) for shape in entry.shapes]:
                    slices.append(slice(cnt, cnt + size))
                    cnt += size
                out_dict[entry.name] = _LinearComponent(True, slices, entry.shapes)
        return cls(**out_dict), cnt

    def _make_input_slices(self) -> tuple[InputSlices, int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            _SliceEntry("zeta_b", True, self.reference.zeta_b.shape),
            _SliceEntry("zeta_b_dot", True, self.reference.zeta_b.shape),
            _SliceEntry(
                "nu_b",
                *(
                    (True, self.reference.zeta_b.shape)
                    if self.bound_upwash
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "nu_w",
                *(
                    (True, self.reference.zeta_w.shape)
                    if self.wake_upwash
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries, InputSlices)

    def _make_state_slices(self) -> tuple[StateSlices, int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            _SliceEntry("gamma_b", True, self.reference.gamma_b.shape),
            _SliceEntry("gamma_w", True, self.reference.gamma_w.shape),
            _SliceEntry(
                "gamma_bm1",
                *(
                    (True, self.reference.gamma_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "gamma_b_dot",
                *(
                    (True, self.reference.gamma_b.shape)
                    if self.gamma_dot_state
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "zeta_w",
                *(
                    (True, self.reference.zeta_w.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "zeta_b",
                *(
                    (True, self.reference.zeta_b.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries, StateSlices)

    def _make_output_slices(self) -> tuple[OutputSlices, int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            _SliceEntry("f_steady", True, self.reference.zeta_b.shape),
            _SliceEntry(
                "f_unsteady",
                *(
                    (True, self.reference.zeta_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries, OutputSlices)

    def _unpack_vector(
        self, x: Array, slices: dict[str, _LinearComponent], add_t: bool = False
    ) -> dict[str, Optional[ArrayList]]:
        r"""
        Unpack a vector into its components based on the provided slices.
        :param x: Vector to unpack, [n_elements] or [n_tstep, n_elements]
        :param slices: Slice name and linear component mapping.
        :param add_t: If true, the first dimension of x_target is time steps.
        :return: Dictionary mapping of names to unpacked ArrayLists.
        """
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                if add_t:
                    n_tstep = x.shape[0]
                    out[name] = ArrayList(
                        [
                            x[:, entry.slices[i_surf]].reshape(
                                n_tstep, *entry.shapes[i_surf]
                            )
                            for i_surf in range(self.reference.n_surf)
                        ]
                    )
                else:
                    out[name] = ArrayList(
                        [
                            x[entry.slices[i_surf]].reshape(entry.shapes[i_surf])
                            for i_surf in range(self.reference.n_surf)
                        ]
                    )
        return out

    def _unpack_input_vector(self, u: Array) -> InputUnflattened:
        r"""
        Unpack an input vector into its components.
        :param u: Input vector, [n_inputs]
        :return: InputUnflattened object.
        """
        return InputUnflattened(
            **self._unpack_vector(u, _shallow_asdict(self.input_slices))
        )

    def _unpack_state_vector(self, x: Array) -> StateUnflattened:
        r"""
        Unpack a state vector into its components.
        :param x: State vector, [n_states]
        :return: StateUnflattened object.
        """
        return StateUnflattened(
            **self._unpack_vector(x, _shallow_asdict(self.state_slices))
        )

    def _unpack_output_vector(self, y: Array) -> OutputUnflattened:
        r"""
        Unpack an output vector into its components.
        :param y: Output vector, [n_outputs]
        :return: OutputUnflattened object.
        """
        return OutputUnflattened(
            **self._unpack_vector(y, _shallow_asdict(self.output_slices))
        )

    def _unpack_input_vector_t(self, u_t: Array) -> InputUnflattened:
        r"""
        Unpack a time history of input vectors into its components.
        :param u_t: Input vector time history, [n_tstep, n_inputs]
        :return: InputUnflattened object.
        """
        return InputUnflattened(
            **self._unpack_vector(u_t, _shallow_asdict(self.input_slices), add_t=True)
        )

    def _unpack_state_vector_t(self, x_t: Array) -> StateUnflattened:
        r"""
        Unpack a time history of state vectors into its components.
        :param x_t: State vector time history, [n_tstep, n_states]
        :return: StateUnflattened object.
        """
        return StateUnflattened(
            **self._unpack_vector(x_t, _shallow_asdict(self.state_slices), add_t=True)
        )

    def _unpack_output_vector_t(self, y_t: Array) -> OutputUnflattened:
        r"""
        Unpack a time history of output vectors into its components.
        :param y_t: Output vector time history, [n_tstep, n_outputs]
        :return: OutputUnflattened object.
        """
        return OutputUnflattened(
            **self._unpack_vector(y_t, _shallow_asdict(self.output_slices), add_t=True)
        )

    def _pack_vector(
        self,
        slices: dict[str, _LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[ArrayList]],
    ) -> Array:
        r"""
        Pack an unflattened object into a vector based on the provided slices.
        :param slices: Mapping of names to linear components.
        :param vec_length: Size of the output vector.
        :param arrs: Mapping of names to ArrayLists to pack.
        :return: Vector, [vec_length]
        """
        vec = jnp.zeros(vec_length)
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.reference.n_surf):
                    vec = vec.at[entry.slices[i_surf]].set(arrs[name][i_surf].ravel())
        return vec

    def _pack_input_vector(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack an input unflattened object into a vector.
        :param u_input: InputUnflattened object.
        :return: Input vector, [n_inputs]
        """
        return self._pack_vector(
            _shallow_asdict(self.input_slices), self.n_inputs, _shallow_asdict(u_input)
        )

    def _pack_state_vector(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a state unflattened object into a vector.
        :param x_state: StateUnflattened object.
        :return: State vector, [n_states]
        """
        return self._pack_vector(
            _shallow_asdict(self.state_slices), self.n_states, _shallow_asdict(x_state)
        )

    def _pack_output_vector(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack an output unflattened object into a vector.
        :param y_output: OutputUnflattened object.
        :return: Output vector, [n_outputs]
        """
        return self._pack_vector(
            _shallow_asdict(self.output_slices),
            self.n_outputs,
            _shallow_asdict(y_output),
        )

    def _pack_vector_t(
        self,
        slices: dict[str, _LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[ArrayList]],
    ) -> Array:
        r"""
        Pack a time history of unflattened objects into a time history of vectors.
        :param slices: Dictionary mapping names to linear components.
        :param vec_length: Length of the output vector.
        :param arrs: Dictionary mapping names to ArrayLists to pack.
        :return: Array, [n_tstep, vec_length]
        """
        n_tstep = list(arrs.values())[0][0].shape[
            0
        ]  # find number of timesteps from first surface, first entry
        vec_t = jnp.zeros((n_tstep, vec_length))
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.reference.n_surf):
                    vec_t = vec_t.at[:, entry.slices[i_surf]].set(
                        arrs[name][i_surf].reshape(n_tstep, -1)
                    )
        return vec_t

    def _pack_input_vector_t(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack a time history of input unflattened objects into a time history of input vectors.
        :param u_input: InputUnflattened object.
        :return: Input vector time history, [n_tstep, n_inputs]
        """
        return self._pack_vector_t(
            _shallow_asdict(self.input_slices), self.n_inputs, _shallow_asdict(u_input)
        )

    def _pack_state_vector_t(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a time history of state unflattened objects into a time history of state vectors.
        :param x_state: StateUnflattened object.
        :return: State vector time history, [n_tstep, n_states]
        """
        return self._pack_vector_t(
            _shallow_asdict(self.state_slices), self.n_states, _shallow_asdict(x_state)
        )

    def _pack_output_vector_t(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack a time history of output unflattened objects into a time history of output vectors.
        :param y_output: OutputUnflattened object.
        :return: Output vector time history, [n_tstep, n_outputs]
        """
        return self._pack_vector_t(
            _shallow_asdict(self.output_slices),
            self.n_outputs,
            _shallow_asdict(y_output),
        )

    def _get_total(
        self,
        input_: dict[str, Optional[ArrayList]],
        reference: dict[str, Optional[ArrayList]],
        add_t: bool = False,
    ) -> dict[str, Optional[ArrayList]]:
        r"""
        Get the total value by adding the reference to the input perturbation.
        :param input_: Dictionary mapping of names to ArrayList perturbation entries
        :param reference: Dictionary mapping of names to ArrayList reference entries
        :param add_t: If true, the first dimension of the arrays is time steps.
        :return: Dictionary mapping of names to total ArrayList entries.
        """
        out = {}
        for name, entry in reference.items():
            if entry is None:
                out[name] = None
            else:
                arrs = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    if add_t:
                        arrs.append(
                            reference[name][i_surf][None, ...] + input_[name][i_surf]
                        )
                    else:
                        arrs.append(reference[name][i_surf] + input_[name][i_surf])
                out[name] = arrs
        return out

    def get_total_input(self, u: InputUnflattened) -> InputUnflattened:
        r"""
        Get the total input by adding the reference to the input perturbation.
        :param u: InputUnflattened perturbation object.
        :return: InputUnflattened total object.
        """
        return InputUnflattened(
            **self._get_total(
                _shallow_asdict(u), _shallow_asdict(self.get_reference_inputs())
            )
        )

    def get_total_state(self, x: StateUnflattened) -> StateUnflattened:
        r"""
        Get the total state by adding the reference to the state perturbation.
        :param x: StateUnflattened perturbation object.
        :return: StateUnflattened total object.
        """
        return StateUnflattened(
            **self._get_total(
                _shallow_asdict(x), _shallow_asdict(self.get_reference_states())
            )
        )

    def get_total_output(self, y: OutputUnflattened) -> OutputUnflattened:
        r"""
        Get the total output by adding the reference to the output perturbation.
        :param y: OutputUnflattened perturbation object.
        :return: OutputUnflattened total object.
        """
        return OutputUnflattened(
            **self._get_total(
                _shallow_asdict(y), _shallow_asdict(self.get_reference_outputs())
            )
        )

    def get_total_input_t(self, u_t: InputUnflattened) -> InputUnflattened:
        r"""
        Get the total input time history by adding the reference to the input perturbation time history.
        :param u_t: InputUnflattened perturbation time history object.
        :return: InputUnflattened total time history object.
        """
        return InputUnflattened(
            **self._get_total(
                _shallow_asdict(u_t),
                _shallow_asdict(self.get_reference_inputs()),
                add_t=True,
            )
        )

    def get_total_state_t(self, x_t: StateUnflattened) -> StateUnflattened:
        r"""
        Get the total state time history by adding the reference to the state perturbation time history.
        :param x_t: StateUnflattened perturbation time history object.
        :return: StateUnflattened total time history object.
        """
        return StateUnflattened(
            **self._get_total(
                _shallow_asdict(x_t),
                _shallow_asdict(self.get_reference_states()),
                add_t=True,
            )
        )

    def get_total_output_t(self, y_t: OutputUnflattened) -> OutputUnflattened:
        r"""
        Get the total output time history by adding the reference to the output perturbation time history.
        :param y_t: OutputUnflattened perturbation time history object.
        :return: OutputUnflattened total time history object.
        """
        return OutputUnflattened(
            **self._get_total(
                _shallow_asdict(y_t),
                _shallow_asdict(self.get_reference_outputs()),
                add_t=True,
            )
        )

    def _get_zero(
        self, slices: dict[str, _LinearComponent]
    ) -> dict[str, Optional[ArrayList]]:
        r"""
        Get a zero unflattened object based on the provided slices.
        :param slices: Dictionary mapping of names to linear components.
        :return: unflattened object with zero arrays.
        """
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                arrs = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    arrs.append(jnp.zeros(entry.shapes[i_surf]))
                out[name] = arrs
        return out

    def get_zero_input(self) -> InputUnflattened:
        r"""
        Get a zero input unflattened object.
        :return: InputUnflattened object with zero arrays.
        """
        return InputUnflattened(**self._get_zero(_shallow_asdict(self.input_slices)))

    def get_zero_state(self) -> StateUnflattened:
        r"""
        Get a zero state unflattened object.
        :return: StateUnflattened object with zero arrays.
        """
        return StateUnflattened(**self._get_zero(_shallow_asdict(self.state_slices)))

    def get_zero_output(self) -> OutputUnflattened:
        r"""
        Get a zero output unflattened object.
        :return: OutputUnflattened object with zero arrays.
        """
        return OutputUnflattened(**self._get_zero(_shallow_asdict(self.output_slices)))

    def _unflatten_subvec(self, vec: Array, component: _LinearComponent) -> ArrayList:
        r"""
        Obtain an ArrayList of arrays from a subvector based on the provided component.
        :param vec: Total vector, [n_elements]
        :param component: LinearComponent defining the slices and arr_list_shapes.
        :return: ArrayList of arrays for each surface for the given component.
        """
        arrs = ArrayList([])
        cnt = 0
        for i_surf in range(self.reference.n_surf):
            size = reduce(mul, component.shapes[i_surf])
            arrs.append(vec[cnt : cnt + size].reshape(component.shapes[i_surf]))
        return arrs

    def linearise(self) -> LinearSystem:
        r"""
        Linearise the UVLM system about the reference state.
        :return: LinearSystem object representing the linearised system.
        """

        def _make_inv_solve_mat(zeta_bs: ArrayList) -> Array:
            r"""
            Gives the matrix :math:`[A(\zeta_c, \zeta_b) \cdot n]^{-1}`
            :param zeta_bs: Bound vertex positions at time=n+1, [n_surf][zeta_m, zeta_n, 3]
            :return: Solve matrix, [m_tot*n_tot, m_tot*n_tot]
            """
            cs = compute_c(zeta_bs)
            ns = compute_nc(zeta_bs)
            aic_sys = compute_aic_solve(
                cs=cs,
                ns=ns,
                zetas_b=self.reference.zeta_b,
                zetas_w=None,
                kernels_b=self.kernels_b,
                kernels_w=None,
                mirror_point=self.mirror_point,
                mirror_normal=self.mirror_normal,
            )

            return jnp.linalg.inv(aic_sys)

        def _make_v_bc(
            zeta_bs: ArrayList,
            zeta_ws: ArrayList,
            gamma_ws: ArrayList,
            zeta_bs_dot: ArrayList,
        ) -> Array:
            r"""
            Boundary condition velocity at collocation points.
            :param zeta_bs: Bound vertex positions at time=n+1, [n_surf][zeta_m, zeta_n, 3]
            :param zeta_ws: Wake vertex positions at time=n+1, [n_surf][zeta_m_star, zeta_n, 3]
            :param gamma_ws: Wake strengths at time=n+1, [n_surf][m_star, n, 3
            :param zeta_bs_dot: Wake vertex velocities at time=n+1, [n_surf][zetas_m, zeta_n, 3]
            :return: Boundary condition velocity at collocation points, [m_tot*n_tot]
            """
            # all values given at time=n+1
            cs = compute_c(zeta_bs)
            cs_dot = compute_c(zeta_bs_dot)

            ns = compute_nc(zeta_bs)

            v_bc = (
                compute_v_ind(
                    cs=cs, zetas=zeta_ws, gammas=gamma_ws, kernels=self.kernels_w
                )
                + self.reference.flowfield.surf_vmap_call(
                    cs, jnp.array(self.reference.t)
                )
                - cs_dot
            )

            return ArrayList.einsum("ijk,ijk->ij", v_bc, ns).flatten()

        def _v_flow(
            x: Array,
            gamma_b: Optional[ArrayList],
            gamma_w: Optional[ArrayList],
            zeta_b: Optional[ArrayList],
            zeta_w: Optional[ArrayList],
        ) -> Array:
            r"""
            Flow velocity at points x_target due to the flowfield and the bound and wake surfaces. Entries of None are replaced
            with the reference value.
            :param x: Points to evaluate flow velocity at, [..., 3]
            :param gamma_b: Bound circulation strengths at t=n+1, [n_surf][m, n, 3]
            :param gamma_w: Wake circulation strengths at t=n+1, [n_surf][m_star, n, 3]
            :param zeta_b: Bound vertex positions at t=n+1, [n_surf][zeta_m, zeta_n, 3]
            :param zeta_w: Wake vertex positions at t=n+1, [n_surf][zeta_m_star, zeta_n, 3]
            :return: Flow velocity at points x_target, [..., 3]
            """
            # sample flowfield
            v_x = self.reference.flowfield.vmap_call(x, jnp.array(self.reference.t))

            # add influence from elements if gamma is provided
            if gamma_b is not None and gamma_w is not None:
                v_x += compute_v_ind(
                    cs=x,
                    zetas=ArrayList([*zeta_b, *zeta_w]),
                    gammas=ArrayList([*gamma_b, *gamma_w]),
                    kernels=[*self.kernels_b, *self.kernels_w],
                )
            return v_x

        def _propagate_linear_wake(
            u_np1: InputUnflattened, x_n: StateUnflattened
        ) -> tuple[Optional[ArrayList], ArrayList]:
            r"""
            Propagate the linear wake from t=n to t=n+1.
            :param u_np1: Inputs at time=n+1
            :param x_n: States at time=n
            :return: Wake grid perturbations and wake circulation perturbations at time=n+1
            """
            u_np1_tot = self.get_total_input(u_np1)
            x_n_tot = self.get_total_state(x_n)

            def _v_wake_prop(x: Array) -> Array:
                # flow is from previous state
                return _v_flow(
                    x,
                    x_n_tot.gamma_b if self.free_wake else None,
                    x_n_tot.gamma_w if self.free_wake else None,
                    x_n_tot.zeta_b if self.free_wake else None,
                    x_n_tot.zeta_w if self.free_wake else None,
                )

            # use wake propagation routines from nonlinear case, as they should be equivalent
            zeta_w_np1_tot, gamma_w_np1_tot = propagate_wake(
                x_n_tot.gamma_b,
                x_n_tot.gamma_w,
                u_np1_tot.zeta_b if self.prescribed_wake else self.reference.zeta_b,
                x_n_tot.zeta_w if self.prescribed_wake else self.reference.zeta_w,
                self.delta_w,
                _v_wake_prop,
                self.dt,
                not self.prescribed_wake,
            )

            # obtain the delta for the linear system
            d_gamma_w_np1 = gamma_w_np1_tot - self.reference.gamma_w
            d_zeta_w_np1 = (
                (zeta_w_np1_tot - self.reference.zeta_w)
                if self.prescribed_wake
                else None
            )

            # add pertubations from input velocities
            # note that we here use the inputs at t=n+1 to convect the wake to t=n+1, as this best suits the linear
            # system structure_dv. For the full nonlinear UVLM, we use the inputs at t=n, which can lead to a discrepancy.
            if self.prescribed_wake and self.wake_upwash:
                d_zeta_w_np1 += u_np1_tot.nu_w * self.dt

            return d_zeta_w_np1, d_gamma_w_np1

        def _get_dn(d_zeta_b: Sequence[Array]) -> Sequence[Array]:
            r"""
            Get the perturbation in n vectors due to perturbations in bound grid positions.
            :param d_zeta_b: Perturbations in bound grid positions at t=n+1, [n_surf][zeta_m, zeta_n, 3]
            :return: Perturbations in n vectors at t=n+1, [n_surf][m, n, 3]
            """
            zeta_b_full = d_zeta_b + self.reference.zeta_b
            n_full = compute_nc(zeta_b_full)
            return n_full - self.reference.nc

        # boundary condition velocity and its derivatives [n_c]
        v_bc0 = _make_v_bc(
            self.reference.zeta_b,
            self.reference.zeta_w,
            self.reference.gamma_w,
            self.reference.zeta_b_dot,
        )

        def d_v_bc_d_zeta_b(d_zeta_b: ArrayList) -> Array:
            r"""
            Obtain the jacobian vector product :math:`\frac{\partial v_{bc}}{\partial \zeta_b} \cdot \delta\zeta_b`
            :param d_zeta_b: Perturbation in bound grid positions at t=n+1, [n_surf][zeta_m, zeta_n, 3]
            :return: Perturbation in boundary condition velocity, [n_c]
            """

            primals, tangents = jax.jvp(
                lambda dzb_: _make_v_bc(
                    zeta_bs=dzb_,
                    zeta_ws=self.reference.zeta_w,
                    gamma_ws=self.reference.gamma_w,
                    zeta_bs_dot=self.reference.zeta_b_dot,
                ),
                [self.reference.zeta_b],
                [d_zeta_b],
            )

            return ArrayList(tangents).flatten()

        def d_v_bc_d_zeta_w(d_zeta_w: ArrayList) -> Array:
            r"""
            Obtain the jacobian vector product :math:`\frac{\partial v_{bc}}{\partial \zeta_w} \cdot \delta\zeta_w`
            :param d_zeta_w: Perturbation in wake grid positions at t=n+1, [n_surf][zeta_m_star, zeta_n, 3]
            :return: Perturbation in boundary condition velocity, [n_c]
            """
            primals, tangents = jax.jvp(
                lambda dzw_: _make_v_bc(
                    zeta_bs=self.reference.zeta_b,
                    zeta_ws=dzw_,
                    gamma_ws=self.reference.gamma_w,
                    zeta_bs_dot=self.reference.zeta_b_dot,
                ),
                [self.reference.zeta_w],
                [d_zeta_w],
            )

            return ArrayList(tangents).flatten()

        def d_v_bc_d_gamma_w(d_gamma_w: ArrayList) -> Array:
            r"""
            Obtain the jacobian vector product :math:`\frac{\partial v_{bc}}{\partial \Gamma_w} \cdot \delta\Gamma_w`
            :param d_gamma_w: Perturbation in wake circulation at t=n+1, [n_surf][m_star, n, 3]
            :return: Perturbation in boundary condition velocity, [n_c]
            """
            primals, tangents = jax.jvp(
                lambda dgw_: _make_v_bc(
                    zeta_bs=self.reference.zeta_b,
                    zeta_ws=self.reference.zeta_w,
                    gamma_ws=dgw_,
                    zeta_bs_dot=self.reference.zeta_b_dot,
                ),
                [self.reference.gamma_w],
                [d_gamma_w],
            )
            return ArrayList(tangents).flatten()

        # solve matrix and its derivative, [n_c, n_c]
        solve_mat0 = _make_inv_solve_mat(self.reference.zeta_b)

        def d_solve_mat_d_zeta_b(d_zeta_b: ArrayList) -> Array:
            r"""
            Obtain the jacobian vector product :math:`\frac{\partial [A(\zeta_c, \zeta_b) \cdot n]^{-1}}{\partial \zeta_b} \cdot \delta\zeta_b`
            :param d_zeta_b: Perturbation in bound grid positions at t=n+1, [n_surf][zeta_m, zeta_n, 3]
            :return: Perturbation in solve matrix, [n_c, n_c]
            """
            primals, tangents = jax.jvp(
                _make_inv_solve_mat, [self.reference.zeta_b], [d_zeta_b]
            )
            return tangents

        @jit
        def _a_func(x_n_vec: Array) -> Array:
            r"""
            State update function for the A matrix in the linear system.
            :param x_n_vec: State vector at time=n, [n_states]
            :return: State vector at time=n+1, [n_states]
            """
            x_n = self._unpack_state_vector(x_n_vec)

            # set previous bound circulation
            d_gamma_bm1_np1 = x_n.gamma_b  # working

            # no contribution to bound grid
            d_zeta_b_np1 = (
                ArrayList(
                    [jnp.zeros(shapes) for shapes in self.state_slices.zeta_b.shapes]
                )
                if self.prescribed_wake
                else None
            )

            # use wake routines to get new wake circulation, factoring in variable discretization
            d_zeta_w_np1, d_gamma_w_np1 = _propagate_linear_wake(
                self.get_zero_input(), x_n
            )

            # influence of states on bound circulation
            d_v_bc = d_v_bc_d_gamma_w(d_gamma_w_np1)

            if self.prescribed_wake:
                d_v_bc += d_v_bc_d_zeta_w(d_zeta_w_np1)

            # resulting bound circulation perturbation
            d_gamma_b_np1 = self._unflatten_subvec(
                -solve_mat0 @ d_v_bc, self.state_slices.gamma_b
            )

            d_gamma_b_dot_np1 = (
                (d_gamma_b_np1 - d_gamma_bm1_np1) / self.dt
                if self.gamma_dot_state
                else None
            )

            state_np1 = StateUnflattened(
                d_gamma_b_np1,
                d_gamma_w_np1,
                d_gamma_bm1_np1,
                d_gamma_b_dot_np1,
                d_zeta_w_np1,
                d_zeta_b_np1,
            )
            return self._pack_state_vector(state_np1)

        @jit
        def _b_func(u_np1_vec: Array) -> Array:
            r"""
            Input to state function for the B matrix in the linear system.
            :param u_np1_vec: Input vector at time=n+1, [n_input]
            :return: State vector at time=n+1, [n_states]
            """
            u_np1 = self._unpack_input_vector(u_np1_vec)

            # pertubations in wake (must be computed first, as they affect d_v_bc and hence d_gamma_b)
            d_zeta_w_np1, d_gamma_w_np1 = _propagate_linear_wake(
                u_np1, self.get_zero_state()
            )

            # influence of grid perturbations on boundary condition velocity
            d_v_bc = d_v_bc_d_zeta_b(u_np1.zeta_b)

            # influence of input-driven wake perturbations on boundary condition velocity
            d_v_bc += d_v_bc_d_gamma_w(d_gamma_w_np1)
            if self.prescribed_wake:
                d_v_bc += d_v_bc_d_zeta_w(d_zeta_w_np1)

            # perturbations in flow and bound grid at zeta_bs
            d_n = _get_dn(u_np1.zeta_b)
            d_zeta_dot_c = compute_c(u_np1.zeta_b_dot)
            zeta0_c_dot = compute_c(self.reference.zeta_b_dot)

            if self.bound_upwash:
                d_nu_c = compute_c(u_np1.nu_b)
                d_v_bc += (
                    ArrayList.einsum(
                        "ijk,ijk->ij", d_nu_c - d_zeta_dot_c, self.reference.nc
                    )
                    + ArrayList.einsum("ijk,ijk->ij", -zeta0_c_dot, d_n)
                ).flatten()
            else:
                d_v_bc += (
                    ArrayList.einsum("ijk,ijk->ij", -d_zeta_dot_c, self.reference.nc)
                    + ArrayList.einsum("ijk,ijk->ij", -zeta0_c_dot, d_n)
                ).flatten()

            d_gamma_b_np1_vec = -solve_mat0 @ d_v_bc

            # pertubations in E matrix [n_c, n_c]
            d_e = d_solve_mat_d_zeta_b(u_np1.zeta_b)
            d_gamma_b_np1_vec -= d_e @ v_bc0

            # pertubations in solve matrix
            d_gamma_b_np1 = self._unflatten_subvec(
                d_gamma_b_np1_vec, self.state_slices.gamma_b
            )

            # pertubations in gamma dot state
            d_gamma_dot_np1 = d_gamma_b_np1 / self.dt if self.gamma_dot_state else None

            state_np1 = StateUnflattened(
                d_gamma_b_np1,
                d_gamma_w_np1,
                ArrayList(
                    [jnp.zeros(shapes) for shapes in self.state_slices.gamma_bm1.shapes]
                )
                if self.unsteady_force
                else None,
                d_gamma_dot_np1,
                d_zeta_w_np1,
                u_np1.zeta_b,
            )

            return self._pack_state_vector(state_np1)

        @jit
        def _c_func(x_n_vec: Array) -> Array:
            r"""
            State to output function for the C matrix in the linear system.
            :param x_n_vec: State vector time=n, [n_state]
            :return: Output vector at time=n, [n_outputs]
            """
            x_n = self._unpack_state_vector(x_n_vec)

            if self.unsteady_force:
                d_gamma_dot_n = (
                    x_n.gamma_b_dot
                    if self.gamma_dot_state
                    else (x_n.gamma_b - x_n.gamma_bm1) / self.dt
                )
                d_f_unsteady_n = self.reference.flowfield.rho * ArrayList(
                    [
                        split_to_vertex(arr, (0, 1))
                        for arr in ArrayList.einsum(
                            "ij,ijk->ijk", d_gamma_dot_n, self.reference.nc
                        )
                    ]
                )
            else:
                d_f_unsteady_n = None

            def steady_forcing_c(
                gamma_b: ArrayList,
                gamma_w: ArrayList,
                zeta_w: Optional[ArrayList] = None,
            ):
                r"""
                Steady forcing at time=n due to perturbations in the states.
                """

                def _v_forcing(x: Array) -> Array:
                    r"""
                    Flow velocity at points x_target due to the flowfield and the bound and wake surfaces.
                    :param x: Points to evaluate flow velocity at, [..., 3]
                    :return: Flow velocity at points x_target, [..., 3]
                    """
                    return _v_flow(
                        x,
                        gamma_b,
                        gamma_w,
                        self.reference.zeta_b,
                        zeta_w if zeta_w is not None else self.reference.zeta_w,
                    )

                return calculate_steady_forcing(
                    zeta_bs=self.reference.zeta_b,
                    zeta_dot_bs=self.reference.zeta_b_dot,
                    gamma_bs=gamma_b,
                    gamma_ws=gamma_w,
                    v_func=_v_forcing,
                    v_inputs=None,
                    rho=self.reference.flowfield.rho,
                )

            # obtain perturbation in steady forces due to states
            d_f_steady_n = jax.jvp(
                steady_forcing_c,
                [self.reference.gamma_b, self.reference.gamma_w, self.reference.zeta_w]
                if self.prescribed_wake
                else [self.reference.gamma_b, self.reference.gamma_w],
                [x_n.gamma_b, x_n.gamma_w, x_n.zeta_w]
                if self.prescribed_wake
                else [x_n.gamma_b, x_n.gamma_w],
            )[1]

            return self._pack_output_vector(
                OutputUnflattened(d_f_steady_n, d_f_unsteady_n)
            )

        @jit
        def _d_func(u_n_vec: Array) -> Array:
            r"""
            Input to output function for the D matrix in the linear system.
            :param u_n_vec: Input vector time=n, [n_input]
            :return: Output vector at time=n, [n_outputs]
            """
            u_n = self._unpack_input_vector(u_n_vec)

            def steady_forcing_d(
                zeta_b: ArrayList, zeta_b_dot: ArrayList, nu_b: Optional[ArrayList]
            ) -> ArrayList:
                def _v_forcing(x: Array) -> Array:
                    r"""
                    Flow velocity at points x_target due to the flowfield and the bound and wake surfaces.
                    :param x: Points to evaluate flow velocity at, [..., 3]
                    :return: Flow velocity at points x_target, [..., 3]
                    """
                    return _v_flow(
                        x,
                        self.reference.gamma_b,
                        self.reference.gamma_w,
                        zeta_b,
                        self.reference.zeta_w,
                    )

                return calculate_steady_forcing(
                    zeta_bs=zeta_b,
                    zeta_dot_bs=zeta_b_dot,
                    gamma_bs=self.reference.gamma_b,
                    gamma_ws=self.reference.gamma_w,
                    v_func=_v_forcing,
                    v_inputs=nu_b,
                    rho=self.reference.flowfield.rho,
                )

            if self.bound_upwash:
                d_f_steady_n = jax.jvp(
                    steady_forcing_d,
                    [
                        self.reference.zeta_b,
                        self.reference.zeta_b_dot,
                        ArrayList.zeros_like(self.reference.zeta_b),
                    ],
                    [u_n.zeta_b, u_n.zeta_b_dot, u_n.nu_b],
                )[1]
            else:
                d_f_steady_n = jax.jvp(
                    lambda zb, zbd: steady_forcing_d(zb, zbd, None),
                    [self.reference.zeta_b, self.reference.zeta_b_dot],
                    [u_n.zeta_b, u_n.zeta_b_dot],
                )[1]

            if self.unsteady_force:
                # no contribution from input_ to unsteady forces, assuming that gamma0_b_dot is zero
                d_f_unsteady_n = ArrayList.zeros_like(d_f_steady_n)
            else:
                d_f_unsteady_n = None

            return self._pack_output_vector(
                OutputUnflattened(d_f_steady_n, d_f_unsteady_n)
            )

        # create linear operators
        a = LinearOperator(_a_func, shape=(self.n_states, self.n_states))
        b = LinearOperator(_b_func, shape=(self.n_states, self.n_inputs))
        c = LinearOperator(_c_func, shape=(self.n_outputs, self.n_states))
        d = LinearOperator(_d_func, shape=(self.n_outputs, self.n_inputs))

        return LinearSystem(a, b, c, d)

    def run(
        self,
        u: InputUnflattened,
        x0: Optional[StateUnflattened] = None,
        flowfield: Optional[FlowField] = None,
        use_matrix=False,
    ) -> AeroLinearResult:
        r"""
        Run the linear system for one time step.
        :param u: Input perturbations over time.
        :param x0: Initial state perturbations, defaults to zero state.
        :param flowfield: FlowField object to provide flow velocities for bound and wake upwash, defaults to no flow.
        :param use_matrix: If true, use explicit matrix representation for linear system, otherwise use operator form.
        """
        if self.prescribed_wake and self.sys.removed_u_np1:
            warn(
                "Wake pertubations coordinates at the trailing edge are zero when removing u_np1 from the system."
            )

        if x0 is None:
            x0_vec = None
        else:
            x0_vec = self._pack_state_vector_t(x0)

        n_tstep: int = u.zeta_b[0].shape[
            0
        ]  # number of time steps from first surface, first entry
        t = self.reference.t + jnp.arange(0, n_tstep) * self.dt  # time vector

        u_tot = u

        if self.bound_upwash and flowfield is None and u_tot.nu_b is None:
            warn(
                "No flowfield or bound upwash perturbations provided. Assuming zero bound upwash perturbations."
            )
            u_tot.nu_b = ArrayList(
                [jnp.zeros((n_tstep, *zb.shape)) for zb in self.reference.zeta_b]
            )

        if self.wake_upwash and flowfield is None and u_tot.nu_w is None:
            warn(
                "No flowfield or wake upwash perturbations provided. Assuming zero wake upwash perturbations."
            )
            u_tot.nu_w = ArrayList(
                [jnp.zeros((n_tstep, *zw.shape)) for zw in self.reference.zeta_w]
            )

        # add flowfield contributions to input upwash if provided
        if flowfield is not None:
            if self.bound_upwash:
                nu_b_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_b_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_b[i_surf], t
                        )
                        - flowfield.vmap_call(self.reference.zeta_b[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_b is None:
                    u_tot.nu_b = nu_b_flow
                else:
                    u_tot.nu_b += nu_b_flow
            if self.wake_upwash:
                nu_w_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_w_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_w[i_surf], t
                        )
                        - flowfield.vmap_call(self.reference.zeta_w[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_w is None:
                    u_tot.nu_w = nu_w_flow
                else:
                    u_tot.nu_w += nu_w_flow
        u_vec = self._pack_input_vector_t(u_tot)

        # run linear system
        x_t, y_t = self.sys.run(u_vec, x0_vec, use_matrix=use_matrix)

        x_t_obj = self._unpack_state_vector_t(x_t)
        y_t_obj = self._unpack_output_vector_t(y_t)
        u_t_tot_obj = self.get_total_input_t(u_tot)
        x_t_tot_obj = self.get_total_state_t(x_t_obj)
        y_t_tot_obj = self.get_total_output_t(y_t_obj)

        # save results to object
        return AeroLinearResult(
            reference=self.reference,
            u_t=u,
            x_t=x_t_obj,
            y_t=y_t_obj,
            u_t_tot=u_t_tot_obj,
            x_t_tot=x_t_tot_obj,
            y_t_tot=y_t_tot_obj,
            n_tstep=n_tstep,
            t=t,
            n_surf=self.reference.n_surf,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
        )

    def eigenvalues(self, to_components: bool = True) -> Array:
        r"""
        Compute stability eigenvalues of the linear system A matrix.
        :param to_components: If true, return real and imaginary parts as separate components. If false, return complex
        eigenvalues.
        :return: Eigenvalues of the system A matrix, [n_states] or [n_states, 2] if to_components is True.
        """
        evals = jnp.linalg.eigvals(self.sys.a.matrix)
        if to_components:
            return jnp.stack((evals.real, evals.imag), axis=-1)
        else:
            return evals

    def reference_snapshot(self) -> DynamicAeroCase:
        r"""
        Get the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :return: StaticAero at reference state
        """
        return DynamicAeroCase(
            zeta_b=self.reference.zeta_b,
            zeta_b_dot=self.reference.zeta_b_dot,
            zeta_w=self.reference.zeta_w,
            gamma_b=self.reference.gamma_b,
            gamma_b_dot=self.reference.gamma_b_dot,
            gamma_w=self.reference.gamma_w,
            f_steady=self.reference.f_steady,
            f_unsteady=self.reference.f_unsteady,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=-1,
            t=jnp.zeros(()),
            n_surf=self.reference.n_surf,
        )

    def plot_reference(
        self, directory: os.PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), plot_wake=plot_wake
        )
