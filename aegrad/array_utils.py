from __future__ import annotations
from jax import Array
from typing import Optional, Sequence, Union
from jax import numpy as jnp
from collections import UserList
from aegrad.utils import make_pytree
from functools import singledispatchmethod


def check_arr_shape(arr: Array, expected_shape: tuple[Optional[int], ...], name: Optional[str]) -> None:
    """Asserts that the shapes of the given array matches the expected shapes.

    Args:
        arr (Array): The array whose shapes is to be checked.
        expected_shape (tuple[Optional[int], ...]): The expected shapes of the array. Values of None allow for any size
        in that dimension.

    Raises:
        ValueError: If the shapes of the array does not match the expected shapes.
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

    message = f"Expected shapes {expected_shape}, but got shapes {actual_shape}."
    if name is not None:
        message += f"Issue with input '{name}'"
    raise ValueError(message)


def check_arr_ndim(arr: Array, expected_ndim: int, name: Optional[str]) -> None:
    """Asserts that the number of dimensions of the given array matches the expected number.

    Args:
        arr (Array): The array whose number of dimensions is to be checked.
        expected_ndim (int): The expected number of dimensions.

    Raises:
        ValueError: If the number of dimensions of the array does not match the expected number.
    """
    actual_ndim = arr.ndim
    if actual_ndim != expected_ndim:
        message = f"Expected {expected_ndim} dimensions, but got {actual_ndim}."
        if name is not None:
            message += f"Issue with input '{name}'"
        raise ValueError(message)


def check_arr_dtype(arr: Array, expected_dtype: type, name: Optional[str]) -> None:
    """Asserts that the data type of the given array matches the expected data type.

    Args:
        arr (Array): The array whose data type is to be checked.
        expected_dtype (type): The expected data type.

    Raises:
        ValueError: If the data type of the array does not match the expected data type.
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
    :param arrs: Double nested sequence of arrays
    :param axes: Axes along which to concatenate the arrays
    :return: Block matrix
    """
    # obtain the number of levels in the nested sequence
    if len(axes) != 2:
        raise ValueError("axes must be a sequence of two integers.")

    return jnp.concatenate(
        [jnp.concatenate(arrs1, axis=axes[1]) for arrs1 in arrs], axis=axes[0]
    )


def neighbour_average(arr: Array, axes: int | Sequence[int]) -> Array:
    r"""
    Average the values of the array along the specified axes.
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


def split_to_vertex(arr: Array, axes: int | Sequence[int]) -> Array:
    r"""
    Split the array into its vertex components along the specified axes. This corresponds to the process of splitting
    the forcing generated by a panel into its four corners.
    :param arr: Input array to split. [..., n_, ..., m_, ...]
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

@make_pytree
class ArrayList(UserList[Array]):
    r"""
    Class to hold a sequence of arrays, useful for handling multiple surfaces.
    :param arrs: Sequence of arrays to hold.
    """

    def __init__(self, arrs: Sequence[Array]) -> None:
        super().__init__(arrs)

    def __add__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] + other[i] for i in range(len(self))])

    def __sub__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] - other[i] for i in range(len(self))])

    def __neg__(self) -> ArrayList:
        return ArrayList([-self[i] for i in range(len(self))])

    def __mul__(self, val: Array | float) -> ArrayList:
        return ArrayList([self[i] * val for i in range(len(self))])

    def __rdiv__(self, val: Array | float) -> ArrayList:
        return ArrayList([self[i] / val for i in range(len(self))])

    def __rmul__(self, val: Array | float) -> ArrayList:
        return self.__mul__(val)

    def __matmul__(self, other: ArrayList) -> ArrayList:
        return ArrayList([self[i] @ other[i] for i in range(len(self))])

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

    def flatten(self) -> Array:
        r"""
        Flatten the sequence of arrays into a single 1D array.
        :return: Flattened 1D array.
        """
        return flatten_to_1d(self)

    def index_all(self, idx: tuple[Union[int, slice, Ellipsis], ...] | slice | int) -> ArrayList:
        r"""
        Get the value of all arrays at the given index. This is equivalent to self[i][idx] for i in range(n).
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

        return ArrayList([jnp.einsum(subscript, *(op[i] for op in operands)) for i in range(n_arrays)])


    def flatten_func(self):
        children = (self.data, )
        aux_data = ()
        return children, aux_data

    @classmethod
    def unflatten_func(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data = children[0]
        return obj
