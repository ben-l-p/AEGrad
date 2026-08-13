from jax import numpy as jnp

from condor.structure import BeamStructure


class TestModal:
    r"""
    Test the mode shapes of a cantilever beam with a uniform cross-section and material properties.
    """

    length = jnp.array(2.0)
    n_nodes = 40
    n_elem = n_nodes - 1
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(0, length, n_nodes))
    y_vect = jnp.zeros((n_elem, 3)).at[:, 1].set(1.0)
    struct = BeamStructure(num_nodes=n_nodes, connectivity=conn, y_vector=y_vect)

    ea, ga, gj, eiy, eiz = 1e6, 1e6, 1e6, 1e6, 1e1
    m_bar, j_bar = 1.0, 1e-6
    k_cs = jnp.diag(jnp.array((ea, ga, ga, gj, eiy, eiz)))

    m_cs = jnp.zeros((6, 6))
    m_cs = m_cs.at[:3, :3].set(m_bar * jnp.eye(3))
    m_cs = m_cs.at[3:, 3:].set(j_bar * jnp.eye(3))

    struct.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)

    @classmethod
    def test_cantilever_bending(cls):
        r"""
        Check the first bending frequency of a cantilever beam against the analytical solution for a uniform beam.
        """

        ref_case = cls.struct.reference_configuration(
            use_f_ext_follower=False,
            use_f_ext_dead=False,
            use_f_aero=False,
            use_f_grav=False,
            prescribed_dofs=tuple(range(6)),
        )

        mode1_freq = (
            3.5156 / (2.0 * jnp.pi * cls.length**2) * jnp.sqrt(cls.eiz / cls.m_bar)
        )
        frequencies, *_ = cls.struct.modal(case=ref_case)

        assert jnp.isclose(frequencies[0], mode1_freq, rtol=1e-3), (
            f"Expected analytical bending frequency {mode1_freq}, got {frequencies[0]}"
        )

    @classmethod
    def test_double_pinned(cls):
        r"""
        Check the low frequency bending modes of a double-pinned beam against the analytical solution for a uniform beam.
        """

        ref_case = cls.struct.reference_configuration(
            use_f_ext_follower=False,
            use_f_ext_dead=False,
            use_f_aero=False,
            use_f_grav=False,
            prescribed_dofs=(
                0,
                1,
                2,
                3,
                4,
                cls.n_nodes * 6 - 6,
                cls.n_nodes * 6 - 5,
                cls.n_nodes * 6 - 4,
                cls.n_nodes * 6 - 3,
                cls.n_nodes * 6 - 2,
            ),
        )

        n = jnp.arange(1, 5)
        mode_frequencies = (
            1.0
            / (2.0 * jnp.pi)
            * (n * jnp.pi / cls.length) ** 2
            * jnp.sqrt(cls.eiz / cls.m_bar)
        )
        frequencies, *_ = cls.struct.modal(case=ref_case)

        assert jnp.allclose(frequencies[: len(n)], mode_frequencies, rtol=5e-3), (
            f"Expected analytical bending frequencies {mode_frequencies}, got {frequencies}"
        )
