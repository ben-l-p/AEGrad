from jax import numpy as jnp
import jax
from jax import Array
from jax.lax import cond
from typing import Callable
from aegrad.aero.constants import EPSILON, R_CUTOFF

type KernelFunction = Callable[[Array, Array], Array]


def biot_savart(x: Array, y: Array) -> Array:
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = r1 / jnp.linalg.norm(r1) - r2 / jnp.linalg.norm(r2)
    return r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) * 4.0 * jnp.pi) * jnp.dot(r0, diff_r)


@jax.custom_jvp
def make_unit_epsilon(r: Array) -> Array:
    return r / jnp.sqrt(jnp.sum(r**2) + EPSILON**2)


@make_unit_epsilon.defjvp
def smooth_unit_vector_jvp(primals, tangents):
    (r,) = primals
    (r_dot,) = tangents
    r_norm2 = jnp.sum(r**2)
    r_norm = jnp.sqrt(r_norm2)

    jvp = jax.lax.select(
        r_norm > R_CUTOFF,
        r_dot / (r_norm + EPSILON)
        - jnp.outer(r, r)
        @ r_dot
        / (jnp.sqrt(r_norm2 + EPSILON**2) * (r_norm + EPSILON) ** 2),
        jnp.zeros(3),
    )

    y = r / (jnp.sqrt(r_norm2 + EPSILON**2))
    return y, jvp


def biot_savart_epsilon(x: Array, y: Array) -> Array:
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = make_unit_epsilon(r1) - make_unit_epsilon(r2)
    r1_x_r2_unit = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) + EPSILON)
    return r1_x_r2_unit / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)


def biot_savart_cutoff(x: Array, y: Array) -> Array:
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]

    sm = jnp.inner(r0, r1) / jnp.inner(r0, y[1, :] - y[0, :])
    m = y[0, :] + sm * (y[1, :] - y[0, :])
    r = jnp.linalg.norm(x - m)  # radial distance

    def _kernel_value() -> Array:
        r1_x_r2 = jnp.cross(r1, r2)
        r1_x_r2_unit2 = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2))
        diff_r = make_unit_epsilon(r1) - make_unit_epsilon(r2)
        return r1_x_r2_unit2 / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)

    return cond((r > R_CUTOFF), _kernel_value, lambda: jnp.zeros(3))
