from __future__ import annotations
from jax import numpy as jnp
from jax import Array
from jax.lax import cond


def matrix2(mat: Array) -> Array:
    r"""
    Computes the square of a matrix.
    :param mat: Matrix, [n, n].
    :return: Matrix squared, [n, n].
    """
    return mat @ mat


def clip_to_pi(val: Array):
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

    # use lower order when not enough data is available
    # for the instance where only a single data point is available, gradient is set to zero
    order = jnp.minimum(order, i_)
    return cond(
        order == 0,
        _order0,
        lambda: cond(order == 1, _order1, lambda: cond(order == 2, _order2, _err)),
    )
