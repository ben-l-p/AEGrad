from __future__ import annotations
from math import factorial
from typing import Callable

import jax
from jax import numpy as jnp, Array
from jax.scipy.special import bernoulli
from jax.lax import cond

from constants import BASE_SUMMATION_ORDER


def matrix2(mat: Array) -> Array:
    r"""
    Computes the square of a matrix.
    :param mat: Matrix, [varphi, varphi].
    :return: Matrix squared, [varphi, varphi].
    """
    return mat @ mat


def clip_to_pi(val: float | Array):
    r"""
    Clips an angle value to be within [-pi, pi].
    :param val: Scalar to bound.
    :return: Bounded scalar within [-pi, pi].
    """
    return jnp.arctan2(jnp.sin(val), jnp.cos(val))


def chi(rmat: Array) -> Array:
    r"""
    Converts a 3x3 rotation matrix to a 6x6 matrix used in spatial transformations.
    :param rmat: Rotation matrix, [a, b].
    :return: Block matrix with diagonal rotation matrices, [2a, 2b].
    """
    return jnp.block([[rmat, jnp.zeros_like(rmat)], [jnp.zeros_like(rmat), rmat]])


def finite_difference(
        i_: int, data: Array, delta: Array, axis: int, order: int = 1
) -> Array:
    r"""
    Compute the finite difference of the data at a given time step. This assumes that data[:i_+1] is available.
    :param i_: Index of derivative to obtain.
    :param data: Data to compute the finite difference on, [...].
    :param delta: Small perturbation value for finite difference, which divides the difference.
    :param axis: Axis along which to compute the finite difference.
    :param order: Order of the finite difference (1 or 2).
    :return: Finite difference of the data at the specified time step.
    """

    if order not in (0, 1, 2):
        raise ValueError("Order must be 0, 1, or 2.")

    def _slice_order(shift_: int) -> tuple[slice | int, ...]:
        sl: list[slice | int] = [slice(None)] * data.ndim
        sl[axis] = i_ - shift_
        return tuple(sl)

    def _order0() -> Array:
        return jnp.zeros([n for i, n in enumerate(data.shape) if i != axis])

    def _order1() -> Array:
        return (data[_slice_order(0)] - data[_slice_order(1)]) / delta

    def _order2() -> Array:
        return (
                3.0 * data[_slice_order(0)]
                - 4.0 * data[_slice_order(1)]
                + data[_slice_order(2)]
        ) / (2.0 * delta)

    def _err() -> Array:
        return jnp.full([n for i, n in enumerate(data.shape) if i != axis], jnp.nan)

    # use lower int_order when not enough data is available
    # for the instance where only a single data point is available, gradient is set to zero
    order: Array = jnp.array((order, i_)).min()
    return cond(
        order == 0,
        _order0,
        lambda: cond(order == 1, _order1, lambda: cond(order == 2, _order2, _err)),
    )


def taylor_series(
        func: Callable[[Array], Array], x0: Array, order: int
) -> Callable[[Array], Array]:
    r"""
    Computes the Taylor series expansion of a function around a point x0 up to a specified order.
    :param func: Function to compute the Taylor series expansion of, which takes an Array and returns an Array.
    :param x0: Point around which to compute the Taylor series expansion, [varphi,].
    :param order: Order of the Taylor series expansion (number of terms to include). Must be a non-negative integer.
    :return: Function that takes an Array and returns the Taylor series expansion of func around x0 up to the
    specified order, [varphi,].
    TODO: This implementation is not efficient for high order Taylor series expansions, as it computes large Jacobian
    matrices. Using the jax.experimental.jet API may be more efficient.
    """

    def get_jvp(delta: Array, order_: int) -> Array:
        curr_func = func
        for _ in range(order_):
            curr_func = jax.jacobian(curr_func)
        mat = curr_func(x0)

        for _ in range(order_):
            mat @= delta
        return mat

    def inner_func(x: Array) -> Array:
        f_x = func(x0)

        for i in range(1, order + 1):
            delta = x - x0
            f_x += get_jvp(delta, i) / factorial(i)
        return f_x

    return inner_func


def exp_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix exponential using truncated summation.
    :param a: Algebra matrix to exponentiate, [varphi, varphi]
    :param order: Order of summation.
    :return: Exponential of matrix, [varphi, varphi]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += jnp.linalg.matrix_power(a, i) / factorial(i)
    return result


def log_sum(g: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the matrix logarithm using truncated summation.
    :param g: Group matrix to exponentiate, [varphi, varphi]
    :param order: Order of summation.
    :return: Logarithm of matrix, [varphi, varphi]
    """

    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("Input must be a square matrix")

    g_e = g - jnp.eye(g.shape[0])
    result = g_e

    for i in range(2, order + 1):
        result += (-1.0) ** (i + 1) * jnp.linalg.matrix_power(g_e, i) / i
    return result


def t_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the tangent operator truncated summation. This is used to validate other implementations.
    :param a: Adjoint action matrix, [varphi, varphi]
    :param order: Order of summation.
    :return: Tangent operator, [varphi, varphi]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * jnp.linalg.matrix_power(a, i) / factorial(i + 1)
    return result


def t_inv_sum(a: Array, order: int = BASE_SUMMATION_ORDER) -> Array:
    r"""
    Computes the inverse tangent operator truncated summation.
    :param a: Adjoint action matrix, [varphi, varphi]
    :param order: Order of summation.
    :return: Inverse angent operator, [varphi, varphi]
    """

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")

    b = bernoulli(order)

    result = jnp.eye(a.shape[0])
    for i in range(1, order + 1):
        result += (-1.0) ** i * b[i] * jnp.linalg.matrix_power(a, i) / factorial(i)
    return result
