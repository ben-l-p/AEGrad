from jax import numpy as jnp
from jax import Array
from typing import Callable, Optional
import equinox


def newton_raphson(
    func: Callable[[Array], tuple[Array, Array]],
    x0: Array,
    atol: float,
    rtol: float,
    n_iter: Optional[int] = 50,
) -> tuple[Array, Array, Array]:
    r"""
    Perform Newton-Raphson iterations to solve for f(x) = 0, where f(x) is some function with known gradient, which may
    be nonlinear.
    :param func: Function that returns the residual and Jacobian given the current guess. [n_dof] -> (residual [n_dof],
    Jacobian [n_dof, n_dof])
    :param x0: Initial guess for the solution, [n_dof]
    :param atol: Absolute tolerance for convergence
    :param rtol: Relative tolerance for convergence
    :param n_iter: Optional maximum number of iterations. If None, will iterate until convergence. If set, may return
    non-converged solution.
    :return: Tuple of converged states [n_dof], residual [n_dof], and Jacobian [n_dof, n_dof]
    """

    f_x0, j_x0 = func(x0)

    # we use the pair (x_{n}, f(x)_{n}, j(x)_n) as the loop variable
    init_state = (x0, f_x0, j_x0)

    def check_convergence(xfj: tuple[Array, Array, Array]) -> Array:
        # returns True if not converged
        f_x = xfj[1]
        abs_norm = jnp.linalg.norm(f_x)
        rel_norm = jnp.linalg.norm(jnp.nan_to_num(f_x / f_x0, False, 0.0))

        return (abs_norm > atol) & (rel_norm > rtol)

    def update(xfj: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x_n, f_x_n, j_x_n = xfj
        x_np1 = x_n - jnp.linalg.solve(j_x_n, f_x_n)  # Newton-Raphson update
        f_x_np1, j_x_np1 = func(
            x_np1
        )  # residual and Jacobian, [n_dof] and [n_dof, n_dof]
        return x_np1, f_x_np1, j_x_np1

    # main loop, return the converged states, residual and Jacobian
    x_conv, f_x_conv, j_x_conv = equinox.internal.while_loop(
        check_convergence,
        update,
        init_state,
        max_steps=n_iter,
        kind="bounded" if n_iter is not None else "lax",
    )
    return x_conv, f_x_conv, j_x_conv
