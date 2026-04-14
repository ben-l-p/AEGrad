from typing import Optional, cast

from jax import numpy as jnp
from jax import Array
import jax

from structure import BeamStructure
from structure import StructureFullStates, StructuralDesignVariables

jax.config.update("jax_enable_x64", True)


class TestLumpedMassTranslationAdjoint:
    @staticmethod
    def test_adjoint():
        n_nodes = 1
        conn = jnp.zeros((0, 2), dtype=int)
        y_vect = jnp.zeros((0, 3))

        beam = BeamStructure(num_nodes=n_nodes, connectivity=conn, y_vector=y_vect,
                             m_lumped_index=jnp.zeros((1,), dtype=int))

        coords = jnp.array(((0.0, 0.0, 0.0),))
        m_l = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))[None, :]

        beam.set_design_variables(
            coords=coords,
            k_cs=jnp.zeros((0, 6, 6)),
            m_cs=jnp.zeros((0, 6, 6)),
            m_lumped=m_l,
        )

        n_tstep = 1001
        dt = 0.001  # results in t in [0, 1]

        f_mag = 2.0

        f_ext = jnp.zeros((n_tstep, n_nodes, 6))
        f_ext = f_ext.at[:, 0, 0].set(f_mag)

        init_state = beam.reference_configuration(use_f_ext_follower=True).to_dynamic()
        init_state.a = init_state.a.at[0, 0].set(f_mag / m_l[0, 0, 0])
        init_state.v_dot = init_state.v_dot.at[0, 0].set(f_mag / m_l[0, 0, 0])
        assert init_state.f_ext_follower is not None, "Initial state has no f_ext_follower"
        init_state.f_ext_follower = init_state.f_ext_follower.at[0, 0].set(f_mag)

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
        x_t_out = solution.hg[:, 0, 0, 3]

        def objective(
                ss: StructureFullStates, _: StructuralDesignVariables, i_ts: Optional[int | Array]
        ) -> Array:
            return jax.lax.select(
                i_ts == n_tstep - 1,
                ss.hg[0, 0, 3],
                jnp.zeros(()),
            )  # x coordinate of point

        # should follow x = 0.5*(f/m)*t^2
        t = jnp.arange(n_tstep, dtype=float) * dt
        expected_x_t = 0.5 * f_mag / m_l[0, 0, 0] * t * t

        grads, adj = beam.dynamic_adjoint(structure=solution, objective=objective)

        f_follower_grad = cast(Array, grads.f_ext_follower)[0, :, 0, 0].sum()
        m_cs_grad = cast(Array, grads.m_lumped)[0, 0, 0, 0]

        expected_f_follower_grad = 0.5 / m_l[0, 0, 0]
        expected_m_cs_grad = -0.5 * f_mag / m_l[0, 0, 0] ** 2

        assert jnp.allclose(expected_x_t, x_t_out), (
            "Primal solution time series does not match analytical solution"
        )

        assert jnp.allclose(f_follower_grad, expected_f_follower_grad), (
            "Follower force gradient does not match analytical solution"
        )
        assert jnp.allclose(m_cs_grad, expected_m_cs_grad), (
            "Mass gradient does not match analytical solution"
        )
