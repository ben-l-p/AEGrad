from __future__ import annotations
import math
from operator import mul
from typing import Optional, Sequence, OrderedDict
from collections import UserList
from functools import singledispatch, reduce
from types import EllipsisType

from jax import numpy as jnp
from jax import Array

from aegrad.utils.utils import make_pytree


def optional_add(*arrs: Optional[Array]) -> Optional[Array]:
    r"""
    Routine to add arrays where some may be None
    :param arrs: Sequence of optional arrays
    :return: Sum of arrays, or None if no passed arrays
    """
    if len(arrs) == 0 or all([arr is None for arr in arrs]):
        return None
    else:
        return sum([arr for arr in arrs if arr is not None])  # type: ignore


def check_arr_shape(
        arr: Array, expected_shape: tuple[Optional[int], ...], name: Optional[str]
) -> None:
    """Asserts that the arr_list_shapes of the given array matches the expected arr_list_shapes.
    :param arr: Input array to check.
    :param expected_shape: Expected shape of the array, as a tuple of integers, with None used for dimensions that can
    be of any size.
    :param name: Name of the input array. This is used to provide more informative error messages.
    :raises ValueError: If the shape of the array does not match the expected shape.
    """
    actual_shape = arr.shape
    if len(actual_shape) == len(expected_shape):
        for i_dim in range(len(expected_shape)):
            if expected_shape[i_dim] is None:
                continue
            if actual_shape[i_dim] != expected_shape[i_dim]:
                break
        else:
            return

    message = f"Expected arr_list_shapes {expected_shape}, but got arr_list_shapes {actual_shape}."
    if name is not None:
        message += f"Issue with input '{name}'"
    raise ValueError(message)


def check_arr_ndim(arr: Array, expected_ndim: int, name: Optional[str]) -> None:
    """Asserts that the number of dimensions of the given array matches the expected number.
    :param arr: Input array to check.
    :param expected_ndim: Expected number of dimensions of the array.
    :param name: Name of the input array. This is used to provide more informative error messages.
    :raises ValueError: If the number of dimensions of the array does not match the expected value.
    """
    actual_ndim = arr.ndim
    if actual_ndim != expected_ndim:
        message = f"Expected {expected_ndim} dimensions, but got {actual_ndim}."
        if name is not None:
            message += f"Issue with input '{name}'"
        raise ValueError(message)


def check_arr_dtype(arr: Array, expected_dtype: type, name: Optional[str]) -> None:
    """Asserts that the data type of the given array matches the expected type.
    :param arr: Input array to check.
    :param expected_dtype: Expected underlying data type of the array.
    :param name: Name of the input array. This is used to provide more informative error messages.
    :raises ValueError: If the data type of the array does not match the expected type.
    """

    if expected_dtype is int:
        jax_dtype = jnp.integer
    elif expected_dtype is float:
        jax_dtype = jnp.floating
    else:
        jax_dtype = expected_dtype

    actual_dtype = arr.dtype
    if not jnp.issubdtype(actual_dtype, jax_dtype):
        message = f"Expected {jax_dtype}, but got {actual_dtype}."
        if name is not None:
            message += f"Issue with input '{name}'"
        raise ValueError(message)


def flatten_to_1d(arrs: Sequence[Array]) -> Array:
    r"""
    Convert a list of ND arrays into a single 1D vector by flattening and concatenating
    :param arrs: List of arrays to flatten and concatenate
    :return: Single 1D vector
    """
    return jnp.concatenate([arr.ravel() for arr in arrs])


def block_axis(arrs: Sequence[Sequence[Array]], axes: Sequence[int]) -> Array:
    r"""
    Form a block matrix along two given axes
    :param arrs: Double nested sequence of arrays [][][varphi, m]
    :param axes: Axes along which to concatenate the arrays
    :return: Block matrix, [n_total, m_total]
    """
    # obtain the number of levels in the nested sequence
    if len(axes) != 2:
        raise ValueError("axes must be a sequence of two integers.")

    return jnp.concatenate(
        [jnp.concatenate(arrs1, axis=axes[1]) for arrs1 in arrs], axis=axes[0]
    )


def neighbour_average(arr: Array, axes: int | Sequence[int]) -> Array:
    r"""
    Find the pairwise average of the array along the specified axes.
    :param arr: Input array to average.
    :param axes: Axis or axes along which to average.
    :return: Averaged array.
    """

    def _single_neighbour_average(arr_: Array, axis_: int) -> Array:
        r"""
        Average the values of the array along the specified axes, considering the neighbouring elements.
        :param arr_: Input array to average.
        :param axis_: Axis along which to average.
        :return: Averaged array.
        """
        index1: list[slice] = [slice(None, None)] * arr_.ndim
        index1[axis_] = slice(None, -1)
        index2: list[slice] = [slice(None, None)] * arr_.ndim
        index2[axis_] = slice(1, None)

        return 0.5 * (arr_[tuple(index1)] + arr_[tuple(index2)])

    if isinstance(axes, int):
        return _single_neighbour_average(arr, axes)
    elif isinstance(axes, Sequence):
        for ax in axes:
            arr = _single_neighbour_average(arr, ax)
        return arr
    else:
        raise TypeError("Axis must be an int or a sequence of ints.")


@make_pytree
class ArrayList(UserList[Array]):
    r"""
    Class to hold a sequence of arrays, with overloaded arithmetic operations. This allows for more elegant handling of
    non-uniform arrays in various calculations.
    :param arrs: Sequence of arrays to hold.
    """

    def __init__(self, arrs: Sequence[Array]) -> None:
        super().__init__(arrs)

    def __add__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] + other[i] for i in range(len(self))])

    def __iadd__(self, other: ArrayList) -> ArrayList:
        for i in range(len(self)):
            self[i] += other[i]
        return self

    def __isub__(self, other) -> ArrayList:
        for i in range(len(self)):
            self[i] -= other[i]
        return self

    def __sub__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] - other[i] for i in range(len(self))])

    def __neg__(self) -> ArrayList:
        return ArrayList([-self[i] for i in range(len(self))])

    def __mul__(self, val: Array | float) -> ArrayList:
        return ArrayList([self[i] * val for i in range(len(self))])

    def __truediv__(self, val: Array | float) -> ArrayList:
        return ArrayList([self[i] / val for i in range(len(self))])

    def __rtruediv__(self, val: Array | float) -> ArrayList:
        return ArrayList([val / self[i] for i in range(len(self))])

    def __rmul__(self, val: Array | float) -> ArrayList:
        return self.__mul__(val)

    def __matmul__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] @ other[i] for i in range(len(self))])

    def to_list(self) -> list[Array]:
        r"""
        Convert the ArrayList to a standard Python list of arrays.
        :return: List of arrays.
        """
        return list(self.data)

    def at(self, idx: int) -> Array:
        r"""
        Get the array at the given index.
        :param idx: Index of the array to get.
        :return: Array at the given index.
        """
        return self.data[idx]

    def combine(self, *other: ArrayList) -> ArrayList:
        new_list = list(self)
        for o_ in other:
            new_list.extend(list(o_))
        return ArrayList(new_list)

    def ravel(self) -> Array:
        r"""
        Flatten the sequence of arrays into a single 1D array.
        :return: Flattened 1D array.
        """
        return flatten_to_1d(self)

    @classmethod
    def from_vector(cls, vect: Array, arr_list_shapes: ArrayListShape) -> ArrayList:
        r"""
        Unravel a 1D vector into a sequence of arrays with the given shapes.
        :param vect: Input 1D vector to unravel.
        :param arr_list_shapes: ArrayListShape containing the shapes of the arrays to unravel into.
        :return: ArrayList containing the unravelled arrays.
        """
        arrs = []
        idx = 0
        for shape in arr_list_shapes.shapes:
            size = math.prod(shape)
            arrs.append(vect[idx: idx + size].reshape(shape))
            idx += size
        return cls(arrs)

    def index_all(
            self,
            *idx: Optional[EllipsisType | int | slice | Array],
    ) -> ArrayList:
        r"""
        Get the value of all arrays at the given index. This is equivalent to self[i][idx] for i in range(varphi).
        """
        return ArrayList([self[i][idx] for i in range(len(self))])

    @staticmethod
    def einsum(subscript: str, *operands: ArrayList) -> ArrayList:
        r"""
        Perform Einstein summation on sequences of arrays.
        :param subscript: Subscript for Einstein summation. This does not include the indices for the sequence dimension.
        :param operands: Sequences of arrays to perform Einstein summation on.
        :return: Sequence of arrays resulting from Einstein summation.
        """
        n_arrays = len(operands[0])
        for op in operands:
            if len(op) != n_arrays:
                raise ValueError("All ArrayLists must have the same length.")

        return ArrayList(
            [
                jnp.einsum(subscript, *(op[i] for op in operands))
                for i in range(n_arrays)
            ]
        )

    @staticmethod
    def zeros_like(arr: ArrayList) -> ArrayList:
        r"""
        Create a new ArrayList with the same shape as the input_, but filled with zeros.
        :param arr: Input ArrayList to create zeros like.
        :return: New ArrayList filled with zeros.
        """
        return ArrayList([jnp.zeros_like(a) for a in arr])

    @property
    def shape(self) -> ArrayListShape:
        r"""
        Get the arr_list_shapes of the arrays in the ArrayList.
        :return: ArrayListShape containing the arr_list_shapes of the arrays in the ArrayList.
        """
        return ArrayListShape(shapes=[arr.shape for arr in self])

    @property
    def size(self) -> int:
        return sum([arr.size for arr in self])

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Get the static field names for the pytree.
        :return: Sequence of static field names.
        """
        return ()

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Get the dynamic field names for the pytree.
        :return: Sequence of dynamic field names.
        """
        return ("data",)


class ArrayListShape:
    r"""
    Class to hold the arr_list_shapes of the arrays in an ArrayList. This is used for indexing and reshaping operations.
    """

    def __init__(self, shapes: Sequence[tuple[int, ...]]) -> None:
        self.shapes: Sequence[tuple[int, ...]] = shapes
        self.n_arrays: int = len(self.shapes)
        self.sizes: Sequence[int] = [reduce(mul, shape, 1) for shape in self.shapes]

    def __iter__(self):
        return iter(self.shapes)

    def __len__(self):
        return self.n_arrays

    def __getitem__(self, i):
        return self.shapes[i]

    def __repr__(self):
        return f"ArrayListShape(n_arrays={self.n_arrays}), shapes={self.shapes}"

    def total_size(self) -> int:
        r"""
        Get the total number of entries in the ArrayList.
        """
        return sum(self.sizes)


@singledispatch
def split_to_vertex(arr: Array, axes: int | Sequence[int]) -> Array:
    r"""
    Split the array into its vertex components along the specified axes. This corresponds to the process of splitting
    the forcing generated by a panel into its four corners.
    :param arr: Input array to split, or equivalent ArrayList. [..., n_, ..., m_, ...]
    :param axes: Axis or axes along which to split.
    :return: Array with vertex components. [..., n_+1, ..., m_+1, ...]
    """

    def _single_split_to_vertex(arr_: Array, axis_: int) -> Array:
        r"""
        Split the array into its vertex components along the specified axis.
        :param arr_: Input array to split.
        :param axis_: Axis along which to split.
        :return: Array with vertex components, with dimension increased by one along the specified axis.
        """

        shape = list(arr_.shape)
        shape[axis_] += 1

        new_arr_ = jnp.empty(shape, dtype=arr_.dtype)

        index1: list[slice] = [slice(None, None)] * len(shape)
        index1[axis_] = slice(None, -1)

        index2: list[slice] = [slice(None, None)] * len(shape)
        index2[axis_] = slice(1, None)

        new_arr_ = new_arr_.at[tuple(index1)].set(0.5 * arr_)
        new_arr_ = new_arr_.at[tuple(index2)].add(0.5 * arr_)
        return new_arr_

    for ax in axes if isinstance(axes, Sequence) else [axes]:
        arr = _single_split_to_vertex(arr, ax)
    return arr


@split_to_vertex.register
def _(arrs: ArrayList, axes: int | Sequence[int]) -> ArrayList:
    return ArrayList([split_to_vertex(arr, axes) for arr in arrs])


def vect_to_arrs(vect: Array, shapes: OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]]) -> OrderedDict[
    str, Optional[Array | ArrayList]]:
    r"""
    Reconstruct a dictionary with a combination of key-Array and key-ArrayList pairs. The shapes of the arrays and array
    lists are specified in the shapes argument, which is an ordered dictionary mapping
    :param vect: Data vector.
    :param shapes: Shapes for unflattened data.
    :return: Ordered dictionary of unflattened data.
    """

    out_vals = OrderedDict()
    cnt: int = 0

    for name, shape in shapes.items():
        if isinstance(shape, tuple):
            sz = reduce(mul, shape, 1)
            out_vals[name] = vect[cnt:cnt + sz].reshape(shape)
            cnt += sz
        elif isinstance(shape, ArrayListShape):
            sz = shape.total_size()
            out_vals[name] = ArrayList.from_vector(vect=vect[cnt:cnt + sz], arr_list_shapes=shape)
            cnt += sz
        elif shape is None:
            out_vals[name] = None
        else:
            raise TypeError("Shape must be a tuple or an ArrayListShape.")
    return out_vals
