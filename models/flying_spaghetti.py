from aegrad.structure.structure import Structure
from jax import Array
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def flying_spaghetti(
    n_nodes: int, t: Array, use_gravity: bool = False
) -> tuple[Structure, Array, Array]:
    r"""
    Creates a flying spaghetti model structure.
    :param n_nodes: Number of nodes.
    :param t: Time array for dead force, [n_tstep]
    :return: Beam model, and dead force array corresponding to the passed time array for respective 2D and 3D cases.
    """

    gay, gaz, ea = 10e4, 10e4, 10e4
    eiy, eiz, gj = 500.0, 500.0, 500.0
    m_bar = 1.0
    j = jnp.array((20.0, 10.0, 10.0))
    y_vector = jnp.array([0.0, 1.0, 0.0])

    k_cs = jnp.diag(jnp.array((ea, gay, gaz, gj, eiy, eiz)))
    m_cs = jnp.diag(jnp.array((m_bar, m_bar, m_bar, *j)))

    n_elem = n_nodes - 1

    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    # beam goes from (6, 0, 0) to (0, 8, 0)
    coords = jnp.zeros((n_nodes, 3))
    coords = coords.at[:, 0].set(jnp.linspace(6.0, 0.0, n_nodes))
    coords = coords.at[:, 2].set(jnp.linspace(0.0, 8.0, n_nodes))

    gravity = jnp.array((0.0, 0.0, -9.81)) if use_gravity else None

    struct = Structure(n_nodes, conn, y_vector, gravity)
    struct.set_design_variables(coords, k_cs, m_cs)

    n_tstep = t.shape[0]

    # g(t) is a signal with g(<=0) = 0, g([0, 2.5]) = 80, g(>=2.5) = 0
    is_active = (t > 0.0) & (t < 2.5)
    g_t_2d = 80.0 * is_active
    f_dead_2d = jnp.zeros((n_tstep, n_nodes, 6))
    f_dead_2d = f_dead_2d.at[:, 0, 0].set(0.1 * g_t_2d)
    f_dead_2d = f_dead_2d.at[:, 0, 4].set(g_t_2d)

    # g(t) is a signal with g(<=0) = 0, g(2.5) = 200, g(>=5) = 0, using linear interpolation

    upper_ramp = 200.0 / 2.5 * t
    lower_ramp = 400.0 - 200.0 / 2.5 * t

    is_upper = (t > 0.0) & (t < 2.5)
    is_lower = (t >= 2.5) & (t < 5.0)

    g_t_3d = upper_ramp * is_upper + lower_ramp * is_lower

    f_dead_3d = jnp.zeros((n_tstep, n_nodes, 6))
    f_dead_3d = f_dead_3d.at[:, 0, 0].set(0.1 * g_t_3d)
    f_dead_3d = f_dead_3d.at[:, 0, 4].set(g_t_3d)
    f_dead_3d = f_dead_3d.at[:, 0, 5].set(0.5 * g_t_3d)

    return struct, f_dead_2d, f_dead_3d


if __name__ == "__main__":
    n_nodes = 21
    dt = 0.02
    t_end = 10.0
    n_tstep = int(jnp.ceil(t_end / dt)) + 1

    # start one timestep behind to prevent issues with initial condition
    t = jnp.arange(n_tstep) * dt - dt

    struct, f_dead_2d, f_dead_3d = flying_spaghetti(n_nodes, t)

    solution = struct.dynamic_solve(
        None,
        n_tstep,
        dt,
        None,
        f_dead_2d,  # swap between 2d and 3d to see the difference in response
        None,
        max_n_iter=40,
        abs_disp_tol=1e-15,
        spectral_radius=0.99,
        include_geometric=True,
        include_material=True,
    )

    # plot_path = Path("./flying_spaghetti_2d")
    # stride = 5
    # solution.plot(plot_path, n_interp=3, index=jnp.arange(0, n_tstep, stride))

    pass
