from typing import cast

from jax import numpy as jnp
from jax import Array
import jax

from aegrad.structure import BeamStructure
from structure import StructureFullStates, StructuralDesignVariables

jax.config.update("jax_enable_x64", True)


class TestBeamTranslationAdjoint:
    @staticmethod
    def test_adjoint():
        n_nodes = 5
        conn = jnp.stack((jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)), axis=1)
        y_vect = jnp.array((0.0, 1.0, 0.0))

        beam = BeamStructure(
            num_nodes=n_nodes, connectivity=conn, y_vector=y_vect[None, :]
        )

        coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(-0.5, 0.5, n_nodes))
        m_cs = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        k_cs = jnp.diag(jnp.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3]))

        beam.set_design_variables(
            coords=coords, k_cs=k_cs[None, ...], m_cs=m_cs[None, ...], m_lumped=None
        )

        n_tstep = 1001
        dt = 0.001  # results in t in [0, 1]

        f_mag = 2.0

        f_ext = jnp.zeros((n_tstep, n_nodes, 6))
        f_ext = f_ext.at[:, 0, 0].set(f_mag)

        init_state = beam.reference_configuration(use_f_ext_follower=True).to_dynamic()
        v_dot_init = jnp.linalg.solve(
            beam.assemble_matrix_from_entries(
                beam.make_m_t(
                    d=jnp.broadcast_to(
                        jnp.array(((1.0 / (n_nodes - 1), 0.0, 0.0, 0.0, 0.0, 0.0),)),
                        (n_nodes - 1, 6),
                    )
                )
            ),
            jnp.zeros(n_nodes * 6).at[0].set(f_mag),
        ).reshape(n_nodes, 6)
        init_state.a = v_dot_init
        init_state.v_dot = v_dot_init
        init_state.f_ext_follower = cast(Array, init_state.f_ext_follower).at[0, 0].set(f_mag)

        solution = beam.dynamic_solve(
            init_state=init_state,
            n_tstep=n_tstep,
            dt=dt,
            f_ext_follower=f_ext,
            f_ext_dead=jnp.zeros_like(f_ext),
            f_ext_aero=None,
            prescribed_dofs=None,
            spectral_radius=1.0,
        )

        # extract x coordinate
        x_t_out = solution.hg[:, :, 0, 3].sum(axis=1) / n_nodes

        def objective(
                ss: StructureFullStates, _: StructuralDesignVariables, i_ts: int | Array | None
        ) -> Array:
            return jax.lax.select(
                i_ts == n_tstep - 1,
                ss.hg[:, 0, 3].sum() / n_nodes,
                jnp.zeros(()),
            )  # x coordinate of point

        # should follow x = 0.5*(f/m)*t^2
        t = jnp.arange(n_tstep, dtype=float) * dt
        expected_x_t = 0.5 * f_mag / m_cs[0, 0] * t * t

        grads, adj = beam.dynamic_adjoint(structure=solution, objective=objective)

        f_follower_grad = cast(Array, grads.f_ext_follower)[0, :, 0, 0].sum()
        m_cs_grad = cast(Array, grads.m_cs)[0, :, 0, 0].sum()

        expected_f_follower_grad = 0.5 / m_cs[0, 0]
        expected_m_cs_grad = -0.5 * f_mag / m_cs[0, 0] ** 2

        assert jnp.allclose(expected_x_t, x_t_out, atol=1e-4), (
            "Primal solution time series does not match analytical solution"
        )

        assert jnp.allclose(f_follower_grad, expected_f_follower_grad, atol=1e-4), (
            "Follower force gradient does not match analytical solution"
        )
        assert jnp.allclose(m_cs_grad, expected_m_cs_grad, atol=1e-4), (
            "Mass gradient does not match analytical solution"
        )
