from aegrad.structure.structure import Structure
from jax import Array
from jax import numpy as jnp
import jax
from pathlib import Path

jax.config.update("jax_enable_x64", True)


def flying_spaghetti(
    n_nodes: int, t: Array, use_gravity: bool = False
) -> tuple[Structure, Array]:
    r"""
    Creates a flying spaghetti model structure.
    :param n_nodes: Number of nodes.
    :param t: Time array for follower force, [n_tstep]
    :return: Beam model, and follower force array corresponding to the passed time array.
    """

    gay, gaz, ea = 10e4, 10e4, 10e4
    eiy, eiz, gj = 50.0, 50.0, 50.0
    m_bar = 1.0
    j_bar = 10.0
    y_vector = jnp.array([0.0, 1.0, 0.0])

    k_cs = jnp.diag(jnp.array((ea, gay, gaz, gj, eiy, eiz)))
    m_cs = jnp.diag(jnp.array((m_bar, m_bar, m_bar, j_bar, j_bar, j_bar)))

    n_elem = n_nodes - 1

    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    # beam goes from (6, 0, 0) to (0, 8, 0)
    coords = jnp.zeros((n_nodes, 3))
    coords = coords.at[:, 0].set(jnp.linspace(6.0, 0.0, n_nodes))
    coords = coords.at[:, 1].set(jnp.linspace(0.0, 8.0, n_nodes))

    gravity = jnp.array((0.0, 0.0, -9.81)) if use_gravity else None

    struct = Structure(n_nodes, conn, y_vector, gravity)
    struct.set_design_variables(coords, k_cs, m_cs)

    # g(t) is a signal with g(<=0) = 0, g(2.5) = 200, g(>=5) = 0, using linear interpolation

    upper_ramp = 200.0 / 2.5 * t
    lower_ramp = 400.0 - 200.0 / 2.5 * t

    is_upper = (t > 0.0) & (t < 2.5)
    is_lower = (t >= 2.5) & (t < 5.0)

    g_t = upper_ramp * is_upper + lower_ramp * is_lower

    n_tstep = t.shape[0]
    f_follower = jnp.zeros((n_tstep, n_nodes, 6))
    f_follower = f_follower.at[:, 0, 0].set(0.1 * g_t)
    f_follower = f_follower.at[:, 0, 1].set(-0.5 * g_t)
    f_follower = f_follower.at[:, 0, 5].set(-g_t)

    return struct, f_follower


if __name__ == "__main__":
    n_nodes = 10
    dt = 0.005
    n_tstep = 1000

    # start one timestep behind to prevent issues with initial condition
    t = jnp.arange(n_tstep) * dt - dt

    struct, f_follower = flying_spaghetti(n_nodes, t)

    f_follower *= 0.1

    solution = struct.dynamic_solve(
        None,
        n_tstep,
        dt,
        f_follower,
        None,
        None,
        max_iter=100,
        abs_tol=1e-15,
        spectral_radius=0.1,
        relaxation_factor=1.0,
    )

    plot_path = Path("./flying_spaghetti")
    solution.plot(plot_path, n_interp=2, index=jnp.arange(0, n_tstep, 50))

    pass
