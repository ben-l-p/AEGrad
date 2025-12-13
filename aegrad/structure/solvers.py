from jax import Array
from typing import Callable
import jax.numpy as jnp


def newton_raphson(
    func: Callable[[Array], Array],
    jac: Callable[[Array], Array],
    x_init: Array,
    free_dof: slice,
) -> Array:
    n_iter = 10

    def update(_, x_km1: Array) -> Array:
        f = func(x_km1)[free_dof]  # [m - n_cnst]
        j = jac(x_km1)[free_dof, free_dof]  # [m - n_cnst, m - n_cnst]
        dx = jnp.linalg.solve(j, f)  # [m - n_cnst]
        return x_km1.at[free_dof].add(-dx)

    # return jax.lax.fori_loop(0, n_iter, update, x_init)
    return update(0, x_init)
