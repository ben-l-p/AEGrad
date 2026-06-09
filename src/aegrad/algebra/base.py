from __future__ import annotations

from time import time
from math import factorial
from typing import Callable, Any, Optional, Sequence

from jax import numpy as jnp, Array
from jax.scipy.special import bernoulli
from jax.lax import cond
from jax import jacrev

from aegrad.utils.constants import BASE_SUMMATION_ORDER


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


def jacrev_kwargs(
    func: Callable[..., Array],
    argnames: str | Sequence[str],
    allow_int: bool = True,
) -> Callable[..., dict[str, Any]]:
    r"""
    Custom reverse Jacobian routine which allows for keyword arguments.
    :param func: Function for which to obtain Jacobians. Must be callable with the keyword arguments later passed to
    the returned function.
    :param argnames: Argument names of variables for which to obtain Jacobians. These must be a subset of the keyword
    argument names provided to the returned function.
    :param allow_int: As `jax.jacrev`.
    :return: Function that accepts keyword arguments and returns a dictionary of argname: Jacobian pairs.
    """

    argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)

    def inner_func(**kwargs: Any) -> dict[str, Array]:
        full_argnames = list(kwargs.keys())
        argvals = list(kwargs.values())
        argnums = tuple(full_argnames.index(name) for name in argnames)

        def positional_adapter(*args: Any) -> Array:
            return func(**dict(zip(full_argnames, args)))

        out_jacs = jacrev(positional_adapter, argnums=argnums, allow_int=allow_int)(
            *argvals
        )
        return dict(zip(argnames, out_jacs))

    return inner_func


def jacrev_custom[T: dict[str, Any]](
    func: Callable[[T], Array],
    args: T,
    jac_options: dict[str, tuple[int, Optional[Callable[[dict[str, Any]], Array]]]],
) -> dict[str, Array]:
    r"""
    Obtain the Jacobians of the function `func` with respect to
    :param func: Function for which to obtain the Jacobians.
    :param args: Full dictionary of arguments required for `func`.
    :param jac_options: Dictionary with variable names as keys, with values being a tuple of the argument number in
    `func`, and an optional function which can be used to compute the given Jacobian. If this is None, the Jacobian is
    computed by applying reverse-mode AD with no approximations.
    :return: Dictionary of argument name - Jacobian array pairs.
    """

    jacobians: dict[str, Array] = {}
    ad_args: dict[
        str, int
    ] = {}  # accumulate dictionary of argument name - number pairs that we will use for AD

    for arg, option in jac_options.items():
        argnum, jac_func = option
        if jac_func is not None:
            jacobians[arg] = jac_func(args)  # evaluate function
        else:
            ad_args.update({arg: argnum})  # indicate that we will use AD to compute

    if ad_args:  # ensure that we don't perform AD when there are no arguments to differentiate for
        jac_func = jacrev_kwargs(func, argnames=list(ad_args.keys()), allow_int=True)
        jacobians.update(jac_func(**args))
    return jacobians


def jacrev_custom_profiling[T: dict[str, Any]](
    func: Callable[[T], Array],
    args: T,
    jac_options: dict[str, tuple[int, Optional[Callable[[dict[str, Any]], Array]]]],
    n_loops: int = 10,
) -> tuple[dict[str, float], dict[str, float]]:

    # TODO: add docstring and console printing

    compile_time: dict[str, float] = {}
    run_time: dict[str, float] = {}

    ad_args: list[
        str
    ] = []  # accumulate dictionary of argument name - number pairs that we will use for AD

    for arg, option in jac_options.items():
        argnum, jac_func = option
        if jac_func is not None:
            # compile
            t_start = time()
            jac_func(args)  # evaluate function
            compile_time[arg] = time() - t_start

            # run
            t_start = time()
            for _ in range(n_loops):
                jac_func(args)
            run_time[arg] = (time() - t_start) / n_loops
        else:
            ad_args.append(arg)  # indicate that we will use AD to compute

    if ad_args:  # ensure that we don't perform AD when there are no arguments to differentiate for
        for ad_arg in ad_args:
            jac_func = jacrev_kwargs(func, argnames=ad_arg, allow_int=True)

            # compile
            t_start = time()
            jac_func(**args)
            run_time[ad_arg] = time() - t_start

            # run
            t_start = time()
            for _ in range(n_loops):
                jac_func(**args)
            run_time[ad_arg] = (time() - t_start) / n_loops

        # case for all gradients at once
        jac_func = jacrev_kwargs(func, argnames=ad_args, allow_int=True)

        # compile
        t_start = time()
        jac_func(**args)
        run_time["all"] = time() - t_start

        # run
        t_start = time()
        for _ in range(n_loops):
            jac_func(**args)
        run_time["all"] = (time() - t_start) / n_loops
    return compile_time, run_time
