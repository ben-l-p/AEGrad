from jax import Array
from jax import numpy as jnp

from condor.structure import BeamStructure


def generate_flying_spaghetti(
    n_nodes: int,
    t: Array,
    use_gravity: bool = False,
    m_lumped: Array | None = None,
    m_bar: Array | None = None,
    spectral_radius: float = 0.9,
) -> tuple[BeamStructure, Array, Array]:
    r"""
    Creates a flying spaghetti model structure.
    :param n_nodes: Number of nodes.
    :param t: Time array for dead force, [n_tstep]
    :param use_gravity: Whether to include gravity in the model.
    :param m_lumped: Optional lumped translational mass per node, [n_nodes].
    :param m_bar: Optional per-element cross-section mass per unit length, [n_elem]. If None, a uniform value of 1.0
        is used.
    :param spectral_radius: Spectral radius for time integrator.
    :return: Beam model, and dead force array corresponding to the passed time array for respective 2D and 3D cases.
    """

    gay, gaz, ea = 10e4, 10e4, 10e4
    eiy, eiz, gj = 500.0, 500.0, 500.0
    j = jnp.array((20.0, 10.0, 10.0))
    y_vector = jnp.array([0.0, 1.0, 0.0])

    n_elem = n_nodes - 1

    k_cs = jnp.diag(jnp.array((ea, gay, gaz, gj, eiy, eiz)))

    if m_bar is None:
        m_cs = jnp.diag(jnp.array((1.0, 1.0, 1.0, *j)))
        m_cs_index = None
    else:
        m_cs = jnp.zeros((n_elem, 6, 6))
        for i in range(3):
            m_cs = m_cs.at[:, i, i].set(m_bar)
        for i, ji in enumerate(j.tolist()):
            m_cs = m_cs.at[:, 3 + i, 3 + i].set(ji)
        m_cs_index = jnp.arange(n_elem)

    if m_lumped is not None:
        m_lump_arr = jnp.zeros((n_nodes, 6, 6))
        for i in range(3):
            m_lump_arr = m_lump_arr.at[:, i, i].set(m_lumped)
        m_lumped_index = jnp.arange(n_nodes)
    else:
        m_lump_arr = None
        m_lumped_index = None

    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    # beam goes from (6, 0, 0) to (0, 8, 0)
    coords = jnp.zeros((n_nodes, 3))
    coords = coords.at[:, 0].set(jnp.linspace(6.0, 0.0, n_nodes))
    coords = coords.at[:, 2].set(jnp.linspace(0.0, 8.0, n_nodes))

    gravity = jnp.array((0.0, 0.0, -9.81)) if use_gravity else None

    struct = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=y_vector,
        gravity=gravity,
        spectral_radius=spectral_radius,
        m_lumped_index=m_lumped_index,
        m_cs_index=m_cs_index,
    )
    struct.set_design_variables(
        coords=coords, k_cs=k_cs, m_cs=m_cs, m_lumped=m_lump_arr
    )

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
