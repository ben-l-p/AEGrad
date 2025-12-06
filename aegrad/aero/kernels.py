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
    return (
            r1_x_r2
            / (jnp.linalg.norm(r1_x_r2) ** 2 * 4.0 * jnp.pi)
            * jnp.dot(r0, diff_r)
    )

@jax.custom_jvp
def make_unit(r: Array) -> Array:
    return r / jnp.sqrt(jnp.sum(r ** 2) + EPSILON ** 2)

@make_unit.defjvp
def smooth_unit_vector_jvp(primals, tangents):
    (r,) = primals
    (r_dot,) = tangents
    r_norm2 = jnp.sum(r**2)
    r_norm = jnp.sqrt(r_norm2)

    jvp = jax.lax.select(r_norm > R_CUTOFF,
                         r_dot / (r_norm + EPSILON) -
                         jnp.outer(r, r) @ r_dot / (jnp.sqrt(r_norm2 + EPSILON ** 2) * (r_norm + EPSILON) ** 2),
                         jnp.zeros(3))

    y = r / (jnp.sqrt(r_norm2 + EPSILON ** 2))
    return y, jvp


def biot_savart_epsilon(x: Array, y: Array) -> Array:
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = make_unit(r1) - make_unit(r2)
    r1_x_r2_unit = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) ** 2 + EPSILON)
    return r1_x_r2_unit / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)

def make_biot_savart_cutoff(cutoff: float = R_CUTOFF) -> KernelFunction:
    def biot_savart_cutoff(x: Array, y: Array) -> Array:
        r0 = y[1, :] - y[0, :]
        r1 = x - y[0, :]
        r2 = x - y[1, :]
        r1_x_r2 = jnp.cross(r1, r2)
        r_norm = jnp.linalg.norm(r1_x_r2)

        def _kernel_value() -> Array:
            diff_r = make_unit(r1) - make_unit(r2)

            return (
                r1_x_r2
                / (r_norm ** 2 * 4.0 * jnp.pi)
                * jnp.dot(r0, diff_r)
            )
        return cond((jnp.linalg.norm(r1_x_r2) > cutoff), _kernel_value, lambda: jnp.zeros(3))
    return biot_savart_cutoff


if __name__ == "__main__":
    zeta = jnp.stack(jnp.meshgrid(jnp.linspace(-1.0, 1.0, 2), jnp.linspace(-1.0, 1.0, 2)), axis=-1)
    zeta = jnp.concatenate((zeta, jnp.zeros_like(zeta[..., [0]])), axis=-1)

    from aegrad.aero.aic import compute_aic_grid
    import jax
    jax.config.update("jax_debug_nans", True)

    # gives defined gradients in both cases A(zeta, c) and A(zeta, zeta)
    # c = neighbour_average(zeta, (0, 1))
    c = zeta

    def _aic(zeta_: Array, c_: Array) -> Array:
        return compute_aic_grid(c_, zeta_, biot_savart_epsilon)

    val = _aic(zeta, c)
    grad_z = jax.jacobian(_aic, argnums=0)(zeta, c)
    grad_c = jax.jacobian(_aic, argnums=1)(zeta, c)

    pass
