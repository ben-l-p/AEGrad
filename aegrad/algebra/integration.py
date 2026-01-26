from jax import Array, vmap
from typing import Callable, Literal
from jax import numpy as jnp


def gauss_lobatto(
    f: Callable[[Array], Array],
    bounds: Array,
    f_bounds: Array,
    int_order: Literal[3, 4, 5],
) -> Array:
    r"""
    Integrate using quadrature with Gauss-Lobatto points. Makes use of function values at the bounds.
    See https://en.wikipedia.org/wiki/Gaussian_quadrature.
    :param f: Function to integrate (must support vector mapping), [1] -> [...]
    :param bounds: Scalar bounds of integration in function space, [2].
    :param f_bounds: values of function at the bounds, [2, ...].
    :param int_order: Order of integration, 3, 4, or 5.
    :return: Integrated value, [...].
    """
    match int_order:
        case 3:
            x_i = jnp.array((0.0,))
            w_i = jnp.array((4.0 / 3.0,))
        case 4:
            x_i = jnp.array((-1.0 / jnp.sqrt(5.0), 1.0 / jnp.sqrt(5.0)))
            w_i = jnp.array((5.0 / 6.0, 5.0 / 6.0))
        case 5:
            x_i = jnp.array((-jnp.sqrt(3.0 / 7.0), 0.0, jnp.sqrt(3.0 / 7.0)))
            w_i = jnp.array((49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0))
        case _:
            raise ValueError("Order must be one of 3, 4, or 5.")

    range = bounds[1] - bounds[0]
    x_i_scaled = bounds[0] + 0.5 * (x_i + 1.0) * range  # [n_i]
    f_i = vmap(f, 0, 0)(x_i_scaled)  # [n_i, ...]

    return (
        range
        / 2.0
        * (
            2.0 / (int_order * (int_order - 1)) * jnp.sum(f_bounds, axis=0)
            + jnp.einsum("i,i...->...", w_i, f_i)
        )
    )
