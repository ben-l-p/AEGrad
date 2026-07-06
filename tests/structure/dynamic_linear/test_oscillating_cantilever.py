from jax import numpy as jnp

from aegrad.structure import BeamStructure


class TestOscillatingCantilever:
    def test_oscillating_undeform_cantilever(self):
        r"""
        Test the oscillating cantilever beam with a tip load. The cantilever begins in an undeformed reference and is
        subject to a vertical tip load, which causes it to oscillate. The test compares the nonlinear and linearised
        solutions of the beam's dynamic response.
        """
        n_nodes = 10
        l = 1.0
        m_bar = 1.0
        j_bar = 0.1
        ea, ga, gj, eay, eaz = 1e4, 1e4, 1e4, 10.0, 10.0

        k_cs = jnp.diag(jnp.array((ea, ga, ga, gj, eay, eaz)))
        m_cs = jnp.diag(jnp.array((m_bar, m_bar, m_bar, j_bar, j_bar, j_bar)))

        coords = jnp.zeros((n_nodes, 3))
        coords = coords.at[:, 0].set(jnp.linspace(0.0, l, n_nodes))

        conn = jnp.zeros((n_nodes - 1, 2), dtype=int)
        conn = conn.at[:, 0].set(jnp.arange(n_nodes - 1))
        conn = conn.at[:, 1].set(jnp.arange(1, n_nodes))

        y_vector = jnp.array((0.0, 1.0, 0.0))

        beam = BeamStructure(
            num_nodes=n_nodes, connectivity=conn, y_vector=y_vector, spectral_radius=1.0
        )

        beam.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)

        ref = beam.reference_configuration(
            use_f_grav=False,
            use_f_aero=False,
            use_f_ext_dead=False,
            use_f_ext_follower=False,
            prescribed_dofs=tuple(range(6)),
        )

        dt = 0.01

        linear_beam = beam.linearise(case=ref, dt=dt)

        # nonlinear case
        physical_time = 2.0
        n_tstep = int(physical_time / dt)
        f_beam = jnp.zeros((n_tstep, n_nodes, 6))
        f_beam = f_beam.at[:, -1, 2].set(1.0)  # tip load

        nl_sol = beam.dynamic_solve(
            init_state=None,
            prescribed_dofs=tuple(range(6)),
            f_ext_dead=f_beam,
            n_tstep=n_tstep,
            dt=dt,
        )

        lin_sol = linear_beam.run(
            f_ext_t=f_beam,
        )

        nl_tip_z = nl_sol.hg[:, -1, 2, 3]
        lin_tip_z = lin_sol.hg[:, -1, 2, 3]

        # chosen tolerances that are considerably smaller than the maximum displacements and rotations
        assert jnp.allclose(nl_tip_z, lin_tip_z, atol=6e-4), (
            "Nonlinear and linear tip displacements do not match"
        )

        nl_tip_rot = nl_sol.varphi[:, -1, 4]
        lin_tip_rot = lin_sol.q[:, -2]

        assert jnp.allclose(nl_tip_rot, lin_tip_rot, atol=8e-4), (
            "Nonlinear and linear tip rotations do not match"
        )
