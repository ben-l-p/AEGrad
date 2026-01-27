from aegrad.print_output import warn
from jax import Array
from jax import numpy as jnp


def get_integration_parameters(
    spectral_radius: float | Array,
    dt: float | Array,
) -> tuple[Array, Array]:
    if 1.0 <= spectral_radius < 0.0:
        warn("Spectral radius should be between 0.0 and 1.0")
    alpha_m = (2.0 * spectral_radius - 1.0) / (spectral_radius + 1.0)
    alpha_f = spectral_radius / (spectral_radius + 1.0)
    gamma = (3.0 - spectral_radius) / (2.0 + 2.0 * spectral_radius)
    beta = 1.0 / (spectral_radius + 1.0) ** 2
    gamma_prime = gamma / (beta * dt)
    beta_prime = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f))
    return jnp.array(gamma_prime), jnp.array(beta_prime)
