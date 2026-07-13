from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, TYPE_CHECKING, Literal
from matplotlib import pyplot as plt

import jax
from jax.scipy.linalg import expm
from jax import Array, numpy as jnp

from aegrad.utils.print_utils import warn, jax_print, VerbosityLevel
from aegrad.utils.utils import shallow_as_dict, make_pytree
from aegrad.algebra.array_utils import (
    ArrayList,
    ArrayListShape,
    check_arr_shape,
)

if TYPE_CHECKING:
    from aegrad.coupled.data_structures import StaticAeroelastic
    from aegrad.aero.data_structures import AeroSnapshot
    from aegrad.structure import StaticStructure
    from aegrad.aero.linear.data_structures import (
        AeroInputUnflattened,
        AeroStateUnflattened,
        AeroOutputUnflattened,
        AeroLinearResult,
    )
    from aegrad.structure.linear.data_structures import (
        BeamInputUnflattened,
        BeamStateUnflattened,
        BeamOutputUnflattened,
        BeamLinearResult,
    )
    from aegrad.coupled.linear.data_structures import (
        AeroelasticInputUnflattened,
        AeroelasticStateUnflattened,
        AeroelasticOutputUnflattened,
        AeroelasticLinearResult,
    )

type InputUnflattened = (
    AeroInputUnflattened | BeamInputUnflattened | AeroelasticInputUnflattened
)
type StateUnflattened = (
    AeroStateUnflattened | BeamStateUnflattened | AeroelasticStateUnflattened
)
type OutputUnflattened = (
    AeroOutputUnflattened | BeamOutputUnflattened | AeroelasticOutputUnflattened
)
type ReferenceObject = StaticStructure | AeroSnapshot | StaticAeroelastic

type LinearResult = AeroLinearResult | BeamLinearResult | AeroelasticLinearResult


@dataclass
class LinearComponent:
    enabled: bool
    slices: Optional[slice | Sequence[slice]]
    shapes: Optional[tuple[int, ...] | ArrayListShape]


@dataclass
class SliceEntry:
    name: str
    enabled: bool
    shapes: Optional[tuple[int, ...] | ArrayListShape]


class LinearModel(ABC):
    r"""
    Class to represent a linearised UVLM aerodynamic system about a reference state.
    """

    def __init__(self, reference: ReferenceObject, dt: float | Array):

        # slices of individual surface components in full vector
        self.input_slices, self.n_inputs = self._make_input_slices(reference=reference)
        self.state_slices, self.n_states = self._make_state_slices(reference=reference)
        self.output_slices, self.n_outputs = self._make_output_slices(
            reference=reference
        )
        self.dt: float | Array = dt

        self._reference: ReferenceObject = reference

        self.reference_inputs: dict[str, Optional[Array | ArrayList]] = (
            self.extract_reference_inputs()
        )
        self.reference_states: dict[str, Optional[Array | ArrayList]] = (
            self.extract_reference_states()
        )
        self.reference_outputs: dict[str, Optional[Array | ArrayList]] = (
            self.extract_reference_outputs()
        )

        self._sys: Optional[LinearSystem] = None

    @property
    def sys(self) -> LinearSystem:
        if self._sys is None:
            raise ValueError("System not linearised")
        return self._sys

    @sys.setter
    def sys(self, sys: LinearSystem):
        self._sys = sys

    @property
    @abstractmethod
    def reference(self) -> ReferenceObject: ...

    @abstractmethod
    def extract_reference_inputs(self) -> dict[str, Optional[Array | ArrayList]]: ...

    @abstractmethod
    def extract_reference_states(self) -> dict[str, Optional[Array | ArrayList]]: ...

    @abstractmethod
    def extract_reference_outputs(self) -> dict[str, Optional[Array | ArrayList]]: ...

    @property
    @abstractmethod
    def input_object(
        self,
    ) -> type: ...

    @property
    @abstractmethod
    def state_object(
        self,
    ) -> type: ...

    @property
    @abstractmethod
    def output_object(
        self,
    ) -> type: ...

    @abstractmethod
    def _make_input_slices(
        self, reference: ReferenceObject
    ) -> tuple[dict[str, LinearComponent], int]: ...

    @abstractmethod
    def _make_state_slices(
        self, reference: ReferenceObject
    ) -> tuple[dict[str, LinearComponent], int]: ...

    @abstractmethod
    def _make_output_slices(
        self, reference: ReferenceObject
    ) -> tuple[dict[str, LinearComponent], int]: ...

    @abstractmethod
    def linearise(self) -> LinearSystem: ...

    def eigenvalues(
        self,
        to_components: bool = False,
        to_continuous_time: bool = True,
        plot: bool = True,
        plot_xlim: tuple[float, float] = (-500.0, 50.0),
        plot_ylim: tuple[float, float] = (-400.0, 400.0),
    ) -> Array:
        r"""
        Compute stability eigenvalues of the linear system A matrix.

        :param to_components: If true, return real and imaginary parts as separate components. If false, return complex
        eigenvalues.
        :param to_continuous_time: If true, convert from discrete to continuous time eigenvalues using the formula:
        lambda_c = log(lambda_d) / dt.
        :param plot: If true, plot the eigenvalues with Matplotlib.
        :param plot_xlim: Range of real component to be used for plotting.
        :param plot_ylim: Range of imaginary component to be used for plotting.
        :return: Eigenvalues of the system A matrix, [n_states] or [n_states, 2] if to_components is True.
        """

        evals = jnp.linalg.eigvals(self.sys.a)

        if to_continuous_time:
            evals = jnp.log(evals) / self.dt

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(
                evals.real,
                evals.imag,
            )
            ax.set_xlim(*plot_xlim)
            ax.set_ylim(*plot_ylim)
            ax.set_xlabel("Re(λ) [1/s]")
            ax.set_ylabel("Im(λ) [1/s]")
            ax.set_title("Eigenvalues")
            plt.show()

        if to_components:
            return jnp.stack((evals.real, evals.imag), axis=-1)
        else:
            return evals

    @abstractmethod
    def run(
        self, u: InputUnflattened, x0: Optional[StateUnflattened] = None
    ) -> LinearResult: ...

    def get_reference_inputs(
        self,
    ) -> InputUnflattened:
        return self.input_object(**self.reference_inputs)

    def get_reference_states(
        self,
    ) -> StateUnflattened:
        return self.state_object(**self.reference_states)

    def get_reference_outputs(
        self,
    ) -> OutputUnflattened:
        return self.output_object(**self.reference_outputs)

    @staticmethod
    def _make_slices(
        slice_entries: Sequence[SliceEntry],
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Helper function to create slices classes for the vectors, and count the number of elements.
        Blocks should be passed in the int_order they are in the dataclass.
        :param slice_entries: Sequence of _SliceEntry objects defining the slices.
        :return: Tuple of (slices class instance, total number of elements).
        """
        # make slices
        cnt = 0
        out_dict = {}
        for entry in slice_entries:
            if not entry.enabled:  # if disabled
                out_dict[entry.name] = LinearComponent(False, None, None)
            else:
                if entry.shapes is None:
                    raise ValueError("Entry.shapes is None")

                if isinstance(entry.shapes, tuple):
                    size: int = prod(entry.shapes)
                    slices = slice(cnt, cnt + size)
                    cnt += size
                elif isinstance(entry.shapes, ArrayListShape):
                    slices: list[slice] = []
                    for shape in entry.shapes:
                        size: int = prod(shape)
                        slices.append(slice(cnt, cnt + size))
                        cnt += size
                else:
                    raise ValueError("Invalid entry.shapes type")
                out_dict[entry.name] = LinearComponent(
                    enabled=True, slices=slices, shapes=entry.shapes
                )
        return out_dict, cnt

    @staticmethod
    def _unpack_vector(
        x: Array, slices: dict[str, LinearComponent], add_t: bool = False
    ) -> dict[str, Optional[Array | ArrayList]]:
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

                if isinstance(entry.slices, slice):
                    if add_t:
                        n_tstep = x.shape[0]
                        out[name] = x[:, entry.slices].reshape(n_tstep, *entry.shapes)
                    else:
                        out[name] = x[entry.slices].reshape(entry.shapes)
                elif isinstance(entry.slices, Sequence):
                    assert isinstance(entry.shapes, ArrayListShape), (
                        "Invalid shape for ArrayList"
                    )
                    if add_t:
                        n_tstep = x.shape[0]
                        out[name] = ArrayList(
                            [
                                x[:, entry.slices[i_surf]].reshape(
                                    n_tstep, *entry.shapes[i_surf]
                                )
                                for i_surf in range(len(entry.slices))
                            ]
                        )
                    else:
                        out[name] = ArrayList(
                            [
                                x[entry.slices[i_surf]].reshape(entry.shapes[i_surf])
                                for i_surf in range(len(entry.slices))
                            ]
                        )
                else:
                    raise ValueError("Invalid slices type")
        return out

    def _unpack_input_vector(self, u: Array) -> object:
        r"""
        Unpack an input vector into its components.
        :param u: Input vector, [n_inputs]
        :return: InputUnflattened object.
        """
        return self.input_object(**self._unpack_vector(x=u, slices=self.input_slices))

    def _unpack_state_vector(self, x: Array) -> object:
        r"""
        Unpack a state vector into its components.
        :param x: State vector, [n_states]
        :return: StateUnflattened object.
        """
        return self.state_object(**self._unpack_vector(x=x, slices=self.state_slices))

    def _unpack_output_vector(self, y: Array) -> object:
        r"""
        Unpack an output vector into its components.
        :param y: Output vector, [n_outputs]
        :return: OutputUnflattened object.
        """
        return self.output_object(**self._unpack_vector(x=y, slices=self.output_slices))

    def _unpack_input_vector_t(self, u_t: Array) -> object:
        r"""
        Unpack a time history of input vectors into its components.
        :param u_t: Input vector time history, [n_tstep, n_inputs]
        :return: InputUnflattened object.
        """
        return self.input_object(
            **self._unpack_vector(x=u_t, slices=self.input_slices, add_t=True)
        )

    def _unpack_state_vector_t(self, x_t: Array) -> object:
        r"""
        Unpack a time history of state vectors into its components.
        :param x_t: State vector time history, [n_tstep, n_states]
        :return: StateUnflattened object.
        """
        return self.state_object(
            **self._unpack_vector(x=x_t, slices=self.state_slices, add_t=True)
        )

    def _unpack_output_vector_t(self, y_t: Array) -> object:
        r"""
        Unpack a time history of output vectors into its components.
        :param y_t: Output vector time history, [n_tstep, n_outputs]
        :return: OutputUnflattened object.
        """
        return self.output_object(
            **self._unpack_vector(x=y_t, slices=self.output_slices, add_t=True)
        )

    @staticmethod
    def _pack_vector(
        slices: dict[str, LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[Array | ArrayList]],
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

                if isinstance(entry.slices, slice):
                    vec = vec.at[entry.slices].set(this_arr.ravel())
                elif isinstance(entry.slices, Sequence):
                    for i_surf, slice_ in enumerate(entry.slices):
                        vec = vec.at[slice_].set(this_arr[i_surf].ravel())
                else:
                    raise ValueError("Invalid slices type")
        return vec

    def _pack_input_vector(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack an input unflattened object into a vector.
        :param u_input: InputUnflattened object.
        :return: Input vector, [n_inputs]
        """
        return self._pack_vector(
            slices=self.input_slices,
            vec_length=self.n_inputs,
            arrs=shallow_as_dict(u_input),
        )

    def _pack_state_vector(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a state unflattened object into a vector.
        :param x_state: StateUnflattened object.
        :return: State vector, [n_states]
        """
        return self._pack_vector(
            slices=self.state_slices,
            vec_length=self.n_states,
            arrs=shallow_as_dict(x_state),
        )

    def _pack_output_vector(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack an output unflattened object into a vector.
        :param y_output: OutputUnflattened object.
        :return: Output vector, [n_outputs]
        """
        return self._pack_vector(
            slices=self.output_slices,
            vec_length=self.n_outputs,
            arrs=shallow_as_dict(y_output),
        )

    @staticmethod
    def _pack_vector_t(
        slices: dict[str, LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[Array | ArrayList]],
    ) -> Array:
        r"""
        Pack a time history of unflattened objects into a time history of vectors.
        :param slices: Dictionary mapping names to linear components.
        :param vec_length: Length of the output vector.
        :param arrs: Dictionary mapping names to ArrayLists to pack.
        :return: Array, [n_tstep, vec_length]
        """

        # find number of timesteps from first surface with valid entry
        n_tstep: int = 0
        for arr in arrs.values():
            if arr is None:
                continue
            if isinstance(arr, Array):
                n_tstep = arr.shape[0]
                break
            elif isinstance(arr, ArrayList):
                n_tstep = arr[0].shape[0]
                break
        if n_tstep == 0:
            raise ValueError("No valid arrays found in arrs")

        vec_t = jnp.zeros((n_tstep, vec_length))
        for name, entry in slices.items():
            if entry.enabled:
                if isinstance(entry.slices, slice):
                    arr = arrs[name]
                    assert isinstance(arr, Array), "Invalid array type"
                    vec_t = vec_t.at[:, entry.slices].set(arr.reshape(n_tstep, -1))
                elif isinstance(entry.slices, Sequence):
                    for i_surf in range(len(entry.slices)):
                        if entry.slices is None:
                            raise ValueError("Invalid shape")
                        if (this_arr := arrs[name]) is None:
                            raise ValueError("Invalid array")
                        vec_t = vec_t.at[:, entry.slices[i_surf]].set(
                            this_arr[i_surf].reshape(n_tstep, -1)
                        )
                else:
                    raise ValueError("Invalid slices type")
        return vec_t

    def _pack_input_vector_t(self, u_input: InputUnflattened) -> Array:
        r"""
        Pack a time history of input unflattened objects into a time history of input vectors.
        :param u_input: InputUnflattened object.
        :return: Input vector time history, [n_tstep, n_inputs]
        """
        return self._pack_vector_t(
            slices=self.input_slices,
            vec_length=self.n_inputs,
            arrs=shallow_as_dict(u_input),
        )

    def _pack_state_vector_t(self, x_state: StateUnflattened) -> Array:
        r"""
        Pack a time history of state unflattened objects into a time history of state vectors.
        :param x_state: StateUnflattened object.
        :return: State vector time history, [n_tstep, n_states]
        """
        return self._pack_vector_t(
            slices=self.state_slices,
            vec_length=self.n_states,
            arrs=shallow_as_dict(x_state),
        )

    def _pack_output_vector_t(self, y_output: OutputUnflattened) -> Array:
        r"""
        Pack a time history of output unflattened objects into a time history of output vectors.
        :param y_output: OutputUnflattened object.
        :return: Output vector time history, [n_tstep, n_outputs]
        """
        return self._pack_vector_t(
            slices=self.output_slices,
            vec_length=self.n_outputs,
            arrs=shallow_as_dict(y_output),
        )

    @staticmethod
    def _get_total(
        input_: dict[str, Optional[Array | ArrayList]],
        reference: dict[str, Optional[Array | ArrayList]],
        add_t: bool = False,
    ) -> dict[str, Optional[Array | ArrayList]]:
        r"""
        Get the total value by adding the reference to the input perturbation.
        :param input_: Dictionary mapping of names to ArrayList perturbation entries
        :param reference: Dictionary mapping of names to ArrayList reference entries
        :param add_t: If true, the first dimension of the arrays is time steps.
        :return: Dictionary mapping of names to total ArrayList entries.
        TODO: some cases are not additive!
        """
        out = {}
        for name, entry in reference.items():
            if entry is None:
                out[name] = None
            else:
                if isinstance(entry, Array):
                    if (this_input := input_[name]) is None:
                        raise ValueError("Invalid input")
                    if add_t:
                        out[name] = entry[None, ...] + this_input
                    else:
                        out[name] = entry + this_input
                elif isinstance(entry, ArrayList):
                    arrs = ArrayList([])
                    for i_surf in range(entry.shape.n_arrays):
                        if (this_input := input_[name]) is None:
                            raise ValueError("Invalid input")
                        if add_t:
                            arrs.append(entry[i_surf][None, ...] + this_input[i_surf])
                        else:
                            arrs.append(entry[i_surf] + this_input[i_surf])
                    out[name] = arrs
                else:
                    raise ValueError("Invalid reference type")
        return out

    def get_total_input(self, u: InputUnflattened) -> InputUnflattened:
        r"""
        Get the total input by adding the reference to the input perturbation.
        :param u: InputUnflattened perturbation object.
        :return: InputUnflattened total object.
        """
        return self.input_object(
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
        return self.state_object(
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
        return self.output_object(
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
        return self.input_object(
            **self._get_total(
                input_=shallow_as_dict(u_t),
                reference=shallow_as_dict(self.get_reference_inputs()),
                add_t=True,
            )
        )

    def get_total_state_t(self, x_t: StateUnflattened) -> StateUnflattened:
        r"""
        Get the total state time history by adding the reference to the state perturbation time history.
        :param x_t: StateUnflattened perturbation time history object.
        :return: StateUnflattened total time history object.
        """
        return self.state_object(
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
        return self.output_object(
            **self._get_total(
                shallow_as_dict(y_t),
                shallow_as_dict(self.get_reference_outputs()),
                add_t=True,
            )
        )

    @staticmethod
    def _get_zero(
        slices: dict[str, LinearComponent],
    ) -> dict[str, Optional[Array | ArrayList]]:
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

                if isinstance(entry.slices, slice):
                    assert isinstance(entry.shapes, tuple)
                    out[name] = jnp.zeros(entry.shapes)
                elif isinstance(entry.slices, Sequence):
                    arrs = ArrayList([])
                    for i_surf, shape in enumerate(entry.shapes):
                        arrs.append(jnp.zeros(shape))
                    out[name] = arrs
        return out

    def get_zero_input(self) -> InputUnflattened:
        r"""
        Get a zero input unflattened object.
        :return: InputUnflattened object with zero arrays.
        """
        return self.input_object(**self._get_zero(shallow_as_dict(self.input_slices)))

    def get_zero_state(self) -> StateUnflattened:
        r"""
        Get a zero state unflattened object.
        :return: StateUnflattened object with zero arrays.
        """
        return self.state_object(**self._get_zero(shallow_as_dict(self.state_slices)))

    def get_zero_output(self) -> OutputUnflattened:
        r"""
        Get a zero output unflattened object.
        :return: OutputUnflattened object with zero arrays.
        """
        return self.output_object(**self._get_zero(shallow_as_dict(self.output_slices)))

    @staticmethod
    def _unflatten_sub_vec(vec: Array, component: LinearComponent) -> Array | ArrayList:
        r"""
        Obtain an ArrayList of arrays from a subvector based on the provided component.
        :param vec: Total vector, [n_elements]
        :param component: LinearComponent defining the slices and arr_list_shapes.
        :return: ArrayList of arrays for each surface for the given component.
        """

        if component.shapes is None:
            raise ValueError("Invalid shape for unflattened object")
        if isinstance(component.shapes, tuple):
            return vec[component.slices].reshape(component.shapes)
        elif isinstance(component.shapes, ArrayListShape):
            arrs = ArrayList([])
            cnt = 0
            for i_surf, shape in enumerate(component.shapes):
                size = prod(shape)
                arrs.append(vec[cnt : cnt + size].reshape(shape))
            return arrs
        else:
            raise ValueError("Invalid shape for unflattened object")


@make_pytree
class LinearSystem:
    r"""
    Linear system represented in state-space form, with tools for time-stepping
    """

    def __init__(
        self,
        a: Array,
        b: Array,
        c: Array,
        d: Array,
        dt: float | Array,
        continuous_time: bool = False,
        removed_u_np1: bool = False,
    ) -> None:
        r"""
        Initialise the LinearSystem with state-space linear operators.
        :param a: System matrix A
        :param b: Input matrix B
        :param c: Output matrix C
        :param d: Feedthrough matrix D
        :param removed_u_np1: If true, indicates that the system is in terms of inputs at time step n only.
        """
        self.a: Array = a
        self.b: Array = b
        self.c: Array = c
        self.d: Array = d
        self.dt: float | Array = dt
        self.n_inputs: int = b.shape[1]
        self.n_states: int = a.shape[0]
        self.n_outputs: int = c.shape[0]
        self.continuous_time: bool = continuous_time
        self.removed_u_np1: bool = removed_u_np1

    def remove_u_np1(self) -> None:
        r"""
        Remove the dependence on u at time varphi+1 from the linear system, modifying b and d accordingly.
        :math:`D_{new} = C B + D` and :math:`B_{new} = A B`
        """
        if self.removed_u_np1:
            warn("u_np1 has already been removed from the system. Skipping.")
        else:
            self.d = (self.c @ self.b) + self.d
            self.b = self.a @ self.b
            self.removed_u_np1 = True

    def continuous_to_discrete(
        self, method: Literal["zoh", "tustin"] = "tustin"
    ) -> LinearSystem:
        r"""
        Convert the continuous-time linear system to a discrete-time linear system.
        """
        jax_print(
            "Converting continuous-time system to discrete-time system.",
            verbose_level=VerbosityLevel.NORMAL,
        )
        if not self.continuous_time:
            warn("System is already discrete-time.")
            return self

        a_c = self.a
        b_c = self.b

        match method:
            case "zoh":
                # perform a ZOH conversion
                # create a matrix aug = [[A, B], [0, 0]], exp(aug * dt) = [[Ad, Bd], [0, I]]
                aug = jnp.block(
                    [
                        [a_c, b_c],
                        [
                            jnp.zeros(
                                (self.b.shape[1], self.a.shape[0] + self.b.shape[1])
                            )
                        ],
                    ]
                )
                assert aug.shape[0] == aug.shape[1], (
                    "Augmented matrix must be square for matrix exponential."
                )

                exp_aug = expm(self.dt * aug)

                a_d = exp_aug[: self.a.shape[0], : self.a.shape[1]]
                b_d = exp_aug[: self.a.shape[0], self.a.shape[1] :]
            case "tustin":
                # Tustin bilinear transformation
                mat_inv = jnp.linalg.inv(jnp.eye(self.a.shape[0]) - a_c * 0.5 * self.dt)
                a_d = mat_inv @ (jnp.eye(self.a.shape[0]) + a_c * 0.5 * self.dt)
                b_d = mat_inv @ (b_c * self.dt)

        return LinearSystem(
            a=a_d, b=b_d, c=self.c, d=self.d, dt=self.dt, continuous_time=False
        )

    def run(self, u: Array, x0: Optional[Array] = None) -> tuple[Array, Array]:
        r"""
        Run the linear system for a time history of input vector u.
        :param u: Time history of input vectors, shape [n_tstep, n_inputs]
        :param x0: Initial state vector, [n_states]. If None, assumed to be zero.
        :return: State history and output history, arr_list_shapes [n_tstep, n_states] and [n_tstep, n_outputs]
        """
        if x0 is not None:
            check_arr_shape(x0, (self.n_states,), "x0")
        check_arr_shape(u, (None, self.n_inputs), "u")
        n_tstep = u.shape[0]

        if self.continuous_time:
            self.continuous_to_discrete()

        def state_func(i_ts: int, x_: Array) -> Array:
            r"""
            State update function for time step i_ts, given as :math:`x_{varphi} = A x_{varphi-1} + B u_{varphi-1}` or :math:`x_n = A x_{varphi-1} + B u_n`.
            :param i_ts: Time step index to obtain new states for.
            :param x_: State history array being updated, [n_tstep, n_states]
            :return: Updated state history array, [n_tstep, n_states]
            """
            jax_print(
                "Linear system state step {i_ts}",
                i_ts=i_ts,
                verbose_level=VerbosityLevel.NORMAL,
            )
            this_u = u[i_ts - 1, ...] if self.removed_u_np1 else u[i_ts, ...]
            return x_.at[i_ts, ...].set(self.a @ x_[i_ts - 1, ...] + self.b @ this_u)

        x = jnp.zeros((n_tstep, self.n_states))
        if x0 is not None:
            x = x.at[0, ...].set(x0)
        x = jax.lax.fori_loop(1, n_tstep, state_func, x)

        def output_func(i_ts: int, y_: Array) -> Array:
            r"""
            Output computation function for time step i_ts, given as :math:`y_n = C x_n + D u_n`.
            :param i_ts: Time step index to obtain outputs for.
            :param y_: Output history array being updated, [n_tstep, n_outputs]
            :return: Updated output history array, [n_tstep, n_outputs]
            """
            jax_print("Linear system output step {i_ts}", i_ts=i_ts)
            return y_.at[i_ts, ...].set(self.c @ x[i_ts, ...] + self.d @ u[i_ts, ...])

        y = jnp.zeros((n_tstep, self.n_outputs))
        y = jax.lax.fori_loop(0, n_tstep, output_func, y)
        return x, y

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Output the names of all static attributes for pytree serialization.
        :return: Sequence of static attribute names.
        """
        return "n_inputs", "n_states", "n_outputs"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Output the names of all dynamic attributes for pytree serialization.
        :return: Sequence of dynamic attribute names.
        """
        return "a", "b", "c", "d", "removed_u_np1"
