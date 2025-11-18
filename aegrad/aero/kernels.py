from jax import numpy as jnp
from jax import Array
from typing import Callable
from aegrad.aero.constants import EPSILON

type KernelFunction = Callable[[Array, Array], Array]

def biot_savart(x: Array, y: Array) -> Array:
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = r1 / jnp.linalg.norm(r1) - r2 / jnp.linalg.norm(r2)
    return (
            r1_x_r2
            / (jnp.linalg.norm(r1_x_r2) ** 2 * 4.0 * jnp.pi)
            * jnp.dot(r0, diff_r)
    )

def make_biot_savart_eps(eps: float = EPSILON) -> KernelFunction:
    def biot_savart_eps(x: Array, y: Array) -> Array:
        r0 = y[1, :] - y[0, :]
        r1 = x - y[0, :]
        r2 = x - y[1, :]
        r1_x_r2 = jnp.cross(r1, r2)
        diff_r = r1 / (jnp.linalg.norm(r1) + eps) - r2 / (
            jnp.linalg.norm(r2) + eps
        )
        return (
            r1_x_r2
            / ((jnp.linalg.norm(r1_x_r2) ** 2 + eps) * 4.0 * jnp.pi)
            * jnp.dot(r0, diff_r)
        )
    return biot_savart_eps