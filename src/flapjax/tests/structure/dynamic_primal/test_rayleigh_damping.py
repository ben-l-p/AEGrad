from jax import numpy as jnp

from flapjax.structure import BeamStructure
from flapjax.structure.linear.data_structures import BeamInputUnflattened


class CantileverBase:
    r"""
    Cantilever used by the Rayleigh damping tests. First bending frequency is well separated
    from the higher modes.
    """

    n_nodes = 8
    length = 1.0
    m_bar = 1.0
    j_bar = 0.05
    ea, ga, gj, eay, eaz = 5e3, 5e3, 5e3, 5.0, 5.0

    k_cs = jnp.diag(jnp.array((ea, ga, ga, gj, eay, eaz)))
    m_cs = jnp.diag(jnp.array((m_bar, m_bar, m_bar, j_bar, j_bar, j_bar)))

    coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(0.0, length, n_nodes))
    conn = (
        jnp.zeros((n_nodes - 1, 2), dtype=int)
        .at[:, 0]
        .set(jnp.arange(n_nodes - 1))
        .at[:, 1]
        .set(jnp.arange(1, n_nodes))
    )
    y_vector = jnp.array((0.0, 1.0, 0.0))

    @classmethod
    def make_beam(cls, alpha_m: float = 0.0, beta_k: float = 0.0) -> BeamStructure:
        beam = BeamStructure(
            num_nodes=cls.n_nodes,
            connectivity=cls.conn,
            y_vector=cls.y_vector,
            spectral_radius=1.0,
            alpha_m=alpha_m,
            beta_k=beta_k,
        )
        beam.set_design_variables(coords=cls.coords, k_cs=cls.k_cs, m_cs=cls.m_cs)
        return beam


class TestRayleighDamping:
    r"""
    Modal analysis should report Rayleigh damping ratios matching the closed-form formula.
    """

    @staticmethod
    def test_modal_damping_ratios() -> None:
        alpha_m = 0.3
        beta_k = 1e-4

        beam_undamped = CantileverBase.make_beam()
        beam_damped = CantileverBase.make_beam(alpha_m=alpha_m, beta_k=beta_k)

        ref = beam_undamped.reference_configuration(
            prescribed_dofs=tuple(range(6)),
        )

        freq_und, damp_und, _ = beam_undamped.modal(case=ref, n_modes=6)
        freq_dmp, damp_dmp, _ = beam_damped.modal(case=ref, n_modes=6)

        # Rayleigh damping does not change the undamped natural frequencies
        assert jnp.allclose(freq_und, freq_dmp, atol=1e-9), (
            "Rayleigh damping should not shift the undamped natural frequencies."
        )

        # Undamped case should report zero damping
        assert jnp.allclose(damp_und, 0.0, atol=1e-12), (
            "Undamped modal analysis should report zero damping ratios."
        )

        omega = 2.0 * jnp.pi * freq_dmp
        expected_zeta = 0.5 * (alpha_m / omega + beta_k * omega)
        assert jnp.allclose(damp_dmp, expected_zeta), (
            "Modal damping ratios do not match the Rayleigh formula."
        )

    @staticmethod
    def test_linear_matches_nonlinear() -> None:
        alpha_m = 0.5
        beta_k = 1e-4
        dt = 0.01
        physical_time = 1.0
        n_tstep = int(physical_time / dt)

        beam = CantileverBase.make_beam(alpha_m=alpha_m, beta_k=beta_k)
        ref = beam.reference_configuration(
            prescribed_dofs=tuple(range(6)),
        )

        linear_beam = beam.linearise(reference=ref, dt=dt, n_modes=None)

        f_beam = jnp.zeros((n_tstep, CantileverBase.n_nodes, 6)).at[:, -1, 2].set(0.3)

        nl_sol = beam.dynamic_solve(
            init_state=None,
            prescribed_dofs=tuple(range(6)),
            f_ext_dead=f_beam,
            f_ext_follower=None,
            n_tstep=n_tstep,
            dt=dt,
        )
        lin_sol = linear_beam.run(u=BeamInputUnflattened(n_tstep=n_tstep, f_ext=f_beam))

        nl_tip_z = nl_sol.x[:, -1, 2]
        lin_tip_z = lin_sol.x[:, -1, 2]

        assert jnp.allclose(nl_tip_z, lin_tip_z, atol=5e-4), (
            "Nonlinear and linear tip deflections diverge with Rayleigh damping active."
        )
