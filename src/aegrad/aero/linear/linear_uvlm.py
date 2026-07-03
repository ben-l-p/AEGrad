from __future__ import annotations

from math import prod
from typing import Sequence, TYPE_CHECKING, Optional
from enum import Enum

from jax import Array, jit, vmap
import jax
import jax.numpy as jnp
import os
from pathlib import Path

from aegrad.aero.data_structures import DynamicAeroCase, AeroSnapshot
from aegrad.aero.gradients.data_structures import AeroStates
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
    calculate_steady_forcing,
)
from aegrad.algebra.linear_operators import LinearOperator, LinearSystem
from aegrad.algebra.array_utils import ArrayList, split_to_vertex
from aegrad.aero.flowfields import FlowField
from aegrad.aero.utils import biot_savart_cutoff, KernelFunction
from aegrad.utils.utils import shallow_as_dict
from aegrad.utils.print_utils import warn
from aegrad.aero.aic import compute_v_ind

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
        Initialise linear UVLM system about a reference state.
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

        # slices of individual surface components in full vector
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
        self.case: UVLM = case

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
                if entry.shapes is None:
                    raise ValueError("Entry.shapes is None")
                slices: list[slice] = []
                for shape in entry.shapes:
                    size: int = prod(shape)
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
                if entry.shapes is None or entry.slices is None:
                    raise ValueError("Invalid shape")
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
            **self._unpack_vector(u, shallow_as_dict(self.input_slices))
        )

    def _unpack_state_vector(self, x: Array) -> StateUnflattened:
        r"""
        Unpack a state vector into its components.
        :param x: State vector, [n_states]
        :return: StateUnflattened object.
        """
        return StateUnflattened(
            **self._unpack_vector(x, shallow_as_dict(self.state_slices))
        )

    def _unpack_output_vector(self, y: Array) -> OutputUnflattened:
        r"""
        Unpack an output vector into its components.
        :param y: Output vector, [n_outputs]
        :return: OutputUnflattened object.
        """
        return OutputUnflattened(
            **self._unpack_vector(y, shallow_as_dict(self.output_slices))
        )

    def _unpack_input_vector_t(self, u_t: Array) -> InputUnflattened:
        r"""
        Unpack a time history of input vectors into its components.
        :param u_t: Input vector time history, [n_tstep, n_inputs]
        :return: InputUnflattened object.
        """
        return InputUnflattened(
            **self._unpack_vector(u_t, shallow_as_dict(self.input_slices), add_t=True)
        )

    def _unpack_state_vector_t(self, x_t: Array) -> StateUnflattened:
        r"""
        Unpack a time history of state vectors into its components.
        :param x_t: State vector time history, [n_tstep, n_states]
        :return: StateUnflattened object.
        """
        return StateUnflattened(
            **self._unpack_vector(x_t, shallow_as_dict(self.state_slices), add_t=True)
        )

    def _unpack_output_vector_t(self, y_t: Array) -> OutputUnflattened:
        r"""
        Unpack a time history of output vectors into its components.
        :param y_t: Output vector time history, [n_tstep, n_outputs]
        :return: OutputUnflattened object.
        """
        return OutputUnflattened(
            **self._unpack_vector(y_t, shallow_as_dict(self.output_slices), add_t=True)
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
                if entry.slices is None:
                    raise ValueError("Invalid shape")
                if (this_arr := arrs[name]) is None:
                    raise ValueError("Invalid array")
                for i_surf in range(self.reference.n_surf):
                    vec = vec.at[entry.slices[i_surf]].set(this_arr[i_surf].ravel())
        return vec

    def _pack_input_vector(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack an input unflattened object into a vector.
        :param u_input: InputUnflattened object.
        :return: Input vector, [n_inputs]
        """
        return self._pack_vector(
            shallow_as_dict(self.input_slices), self.n_inputs, shallow_as_dict(u_input)
        )

    def _pack_state_vector(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a state unflattened object into a vector.
        :param x_state: StateUnflattened object.
        :return: State vector, [n_states]
        """
        return self._pack_vector(
            shallow_as_dict(self.state_slices), self.n_states, shallow_as_dict(x_state)
        )

    def _pack_output_vector(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack an output unflattened object into a vector.
        :param y_output: OutputUnflattened object.
        :return: Output vector, [n_outputs]
        """
        return self._pack_vector(
            shallow_as_dict(self.output_slices),
            self.n_outputs,
            shallow_as_dict(y_output),
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

        # find number of timesteps from first surface with valid entry
        n_tstep: int = [arr[0].shape[0] for arr in arrs.values() if arr is not None][0]

        vec_t = jnp.zeros((n_tstep, vec_length))
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.reference.n_surf):
                    if entry.slices is None:
                        raise ValueError("Invalid shape")
                    if (this_arr := arrs[name]) is None:
                        raise ValueError("Invalid array")
                    vec_t = vec_t.at[:, entry.slices[i_surf]].set(
                        this_arr[i_surf].reshape(n_tstep, -1)
                    )
        return vec_t

    def _pack_input_vector_t(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack a time history of input unflattened objects into a time history of input vectors.
        :param u_input: InputUnflattened object.
        :return: Input vector time history, [n_tstep, n_inputs]
        """
        return self._pack_vector_t(
            shallow_as_dict(self.input_slices), self.n_inputs, shallow_as_dict(u_input)
        )

    def _pack_state_vector_t(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a time history of state unflattened objects into a time history of state vectors.
        :param x_state: StateUnflattened object.
        :return: State vector time history, [n_tstep, n_states]
        """
        return self._pack_vector_t(
            shallow_as_dict(self.state_slices), self.n_states, shallow_as_dict(x_state)
        )

    def _pack_output_vector_t(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack a time history of output unflattened objects into a time history of output vectors.
        :param y_output: OutputUnflattened object.
        :return: Output vector time history, [n_tstep, n_outputs]
        """
        return self._pack_vector_t(
            shallow_as_dict(self.output_slices),
            self.n_outputs,
            shallow_as_dict(y_output),
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
                    if (this_input := input_[name]) is None:
                        raise ValueError("Invalid input")
                    if add_t:
                        arrs.append(entry[i_surf][None, ...] + this_input[i_surf])
                    else:
                        arrs.append(entry[i_surf] + this_input[i_surf])
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
                shallow_as_dict(u), shallow_as_dict(self.get_reference_inputs())
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
                shallow_as_dict(x), shallow_as_dict(self.get_reference_states())
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
                shallow_as_dict(y), shallow_as_dict(self.get_reference_outputs())
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
                shallow_as_dict(u_t),
                shallow_as_dict(self.get_reference_inputs()),
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
                shallow_as_dict(x_t),
                shallow_as_dict(self.get_reference_states()),
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
                shallow_as_dict(y_t),
                shallow_as_dict(self.get_reference_outputs()),
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
                if entry.shapes is None:
                    raise ValueError("Invalid shape for unflattened object")
                out[name] = entry
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
        return InputUnflattened(**self._get_zero(shallow_as_dict(self.input_slices)))

    def get_zero_state(self) -> StateUnflattened:
        r"""
        Get a zero state unflattened object.
        :return: StateUnflattened object with zero arrays.
        """
        return StateUnflattened(**self._get_zero(shallow_as_dict(self.state_slices)))

    def get_zero_output(self) -> OutputUnflattened:
        r"""
        Get a zero output unflattened object.
        :return: OutputUnflattened object with zero arrays.
        """
        return OutputUnflattened(**self._get_zero(shallow_as_dict(self.output_slices)))

    def _unflatten_sub_vec(self, vec: Array, component: _LinearComponent) -> ArrayList:
        r"""
        Obtain an ArrayList of arrays from a subvector based on the provided component.
        :param vec: Total vector, [n_elements]
        :param component: LinearComponent defining the slices and arr_list_shapes.
        :return: ArrayList of arrays for each surface for the given component.
        """
        arrs = ArrayList([])
        cnt = 0
        if component.shapes is None:
            raise ValueError("Invalid shape for unflattened object")
        for i_surf in range(self.reference.n_surf):
            size = prod(component.shapes[i_surf])
            arrs.append(vec[cnt : cnt + size].reshape(component.shapes[i_surf]))
        return arrs

    def _f_step(self, x_vec: Array, u_vec: Array) -> tuple[Array, Array]:
        r"""
        Combined state/output step in vector form, operating on total (reference + perturbation) quantities and
        returning perturbations relative to the reference.
        :param x_vec: State vector, [n_states]
        :param u_vec: Input vector, [n_inputs]
        :return: Tuple of (state perturbation vector, output perturbation vector).
        """
        ref = self.reference
        x = self._unpack_state_vector(x_vec)
        u = self._unpack_input_vector(u_vec)

        # total quantities
        zeta_b_tot = ArrayList([zr + du for zr, du in zip(ref.zeta_b, u.zeta_b)])
        zeta_b_dot_tot = ArrayList(
            [zr + du for zr, du in zip(ref.zeta_b_dot, u.zeta_b_dot)]
        )

        gamma_b_tot = ArrayList([gr + dg for gr, dg in zip(ref.gamma_b, x.gamma_b)])
        gamma_w_tot = ArrayList([gr + dg for gr, dg in zip(ref.gamma_w, x.gamma_w)])
        gamma_b_dot_tot = (
            ArrayList([gr + dg for gr, dg in zip(ref.gamma_b_dot, x.gamma_b_dot)])
            if x.gamma_b_dot is not None
            else ref.gamma_b_dot
        )
        zeta_w_tot = (
            ArrayList([zr + dz for zr, dz in zip(ref.zeta_w, x.zeta_w)])
            if x.zeta_w is not None
            else ref.zeta_w
        )

        q_nm1 = AeroStates(
            gamma_b=gamma_b_tot,
            gamma_w=gamma_w_tot,
            gamma_b_dot=gamma_b_dot_tot,
            zeta_w=zeta_w_tot,
        )
        (
            _,
            _,
            gamma_b_np1,
            gamma_w_np1,
            gamma_b_dot_np1,
            _,
            zeta_w_np1,
            _,
            _,
            _,
        ) = self.case.base_solve_from_grid(
            q_nm1=q_nm1,
            t_n=ref.t,
            zeta_b_n=zeta_b_tot,
            zeta_b_nm1=ref.zeta_b,
            zeta_b_dot_n=zeta_b_dot_tot,
            static=False,
            horseshoe=False,
            linearise_variable_wake=True,
            nu_b=u.nu_b if self.bound_upwash else None,
            nu_w=u.nu_w if (self.prescribed_wake and self.wake_upwash) else None,
        )

        rho = ref.flowfield.rho

        def v_out_func(x_target: Array) -> Array:
            return ref.flowfield.vmap_call(x=x_target, t=ref.t) + compute_v_ind(
                cs=x_target,
                zetas=ArrayList([*zeta_b_tot, *zeta_w_tot]),
                gammas=ArrayList([*gamma_b_tot, *gamma_w_tot]),
                kernels=[*self.kernels_b, *self.kernels_w],
                batch_size=self.case.batch_size,
                mirror_normal=self.case.mirror_normal,
                mirror_point=self.case.mirror_point,
            )

        f_steady_out = calculate_steady_forcing(
            zeta_b=zeta_b_tot,
            zeta_dot_b=zeta_b_dot_tot,
            gamma_b=gamma_b_tot,
            gamma_w=gamma_w_tot,
            rho=rho,
            v_func=v_out_func,
            v_inputs=u.nu_b if self.bound_upwash else None,
        )

        if self.unsteady_force:
            if self.gamma_dot_state:
                assert x.gamma_b_dot is not None
                gamma_b_dot_at_n = ArrayList(
                    [gr + dg for gr, dg in zip(ref.gamma_b_dot, x.gamma_b_dot)]
                )
            else:
                assert x.gamma_bm1 is not None
                gamma_b_dot_at_n = ArrayList(
                    [(gn - gm1) / self.dt for gn, gm1 in zip(x.gamma_b, x.gamma_bm1)]
                )
            f_unsteady_out = ArrayList(
                [
                    split_to_vertex(
                        rho * gamma_b_dot_at_n[i][..., None] * ref.nc[i], (0, 1)
                    )
                    for i in range(ref.n_surf)
                ]
            )
        else:
            f_unsteady_out = None

        # pack perturbations
        d_gamma_b = ArrayList([gn - gr for gn, gr in zip(gamma_b_np1, ref.gamma_b)])
        d_gamma_w = ArrayList([gn - gr for gn, gr in zip(gamma_w_np1, ref.gamma_w)])
        d_zeta_w = (
            ArrayList([zn - zr for zn, zr in zip(zeta_w_np1, ref.zeta_w)])
            if self.prescribed_wake
            else None
        )

        assert gamma_b_dot_np1 is not None
        d_gamma_bm1 = x.gamma_b if self.unsteady_force else None
        d_gamma_b_dot = (
            ArrayList([gn - gr for gn, gr in zip(gamma_b_dot_np1, ref.gamma_b_dot)])
            if self.gamma_dot_state
            else None
        )
        d_zeta_b_state = u.zeta_b if self.prescribed_wake else None

        d_f_steady = ArrayList([fn - fr for fn, fr in zip(f_steady_out, ref.f_steady)])
        if self.unsteady_force:
            assert f_unsteady_out is not None and ref.f_unsteady is not None
            d_f_unsteady = ArrayList(
                [fn - fr for fn, fr in zip(f_unsteady_out, ref.f_unsteady)]
            )
        else:
            d_f_unsteady = None

        state_new = StateUnflattened(
            gamma_b=d_gamma_b,
            gamma_w=d_gamma_w,
            gamma_bm1=d_gamma_bm1,
            gamma_b_dot=d_gamma_b_dot,
            zeta_w=d_zeta_w,
            zeta_b=d_zeta_b_state,
        )
        out_new = OutputUnflattened(f_steady=d_f_steady, f_unsteady=d_f_unsteady)

        return self._pack_state_vector(state_new), self._pack_output_vector(out_new)

    def linearise(self) -> LinearSystem:
        r"""
        Build the linear state-space system.
        """
        x_ref = jnp.zeros(self.n_states)
        u_ref = jnp.zeros(self.n_inputs)

        _, f_lin = jax.linearize(self._f_step, x_ref, u_ref)

        def a_func(dx: Array) -> Array:
            return f_lin(dx, jnp.zeros_like(u_ref))[0]

        def b_func(du: Array) -> Array:
            return f_lin(jnp.zeros_like(x_ref), du)[0]

        def c_func(dx: Array) -> Array:
            return f_lin(dx, jnp.zeros_like(u_ref))[1]

        def d_func(du: Array) -> Array:
            return f_lin(jnp.zeros_like(x_ref), du)[1]

        a = LinearOperator(jit(a_func), shape=(self.n_states, self.n_states))
        b = LinearOperator(jit(b_func), shape=(self.n_states, self.n_inputs))
        c = LinearOperator(jit(c_func), shape=(self.n_outputs, self.n_states))
        d = LinearOperator(jit(d_func), shape=(self.n_outputs, self.n_inputs))

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
                "Wake perturbations coordinates at the trailing edge are zero when removing u_np1 from the system."
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
                            self.reference.zeta_b[i_surf],
                            t,  # type: ignore
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
                            self.reference.zeta_w[i_surf],
                            t,  # type: ignore
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
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
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
        Get the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
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
            cs_ang=self.reference.cs_ang,
            cs_vel=self.reference.cs_vel,
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
            i_ts=jnp.atleast_1d(-1),
            t=jnp.zeros(()),
            c=self.reference.c,
            n=self.reference.nc,
            kernels=self.reference.kernels,
            mirror_normal=self.reference.mirror_normal,
            mirror_point=self.reference.mirror_point,
            flowfield=self.reference.flowfield,
            dof_mapping=self.reference.dof_mapping,
            free_wake=self.reference.free_wake,
            gamma_dot_relaxation=self.reference.gamma_dot_relaxation,
            static_horseshoe=self.reference.static_horseshoe,
            batch_size=self.case.batch_size,
        )

    def plot_reference(
        self, directory: os.PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), index=None, plot_wake=plot_wake
        )
