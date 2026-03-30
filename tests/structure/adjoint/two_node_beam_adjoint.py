from jax import numpy as jnp
from jax import Array
import jax
import numpy as np

from aegrad.structure import BeamStructure
from structure import StructureFullStates, StructuralDesignVariables

jax.config.update("jax_enable_x64", True)


class TestTwoNodeBeamTranslationAdjoint:
    @staticmethod
    def test_adjoint():
        n_nodes = 2
        conn = jnp.array(((0, 1),), dtype=int)
        y_vect = jnp.array((0.0, 1.0, 0.0))

        beam = BeamStructure(num_nodes=n_nodes, connectivity=conn, y_vector=y_vect)

        coords = jnp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        k_cs = jnp.diag(jnp.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3]))[None, :]
        m_cs = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))[None, :]

        beam.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)

        n_tstep = 1001
        dt = 0.001

        f_mag = 2.0

        f_ext = jnp.zeros((n_tstep, n_nodes, 6))
        f_ext = f_ext.at[:, 0, 0].set(f_mag)

        solution = beam.dynamic_solve(
            init_state=None,
            n_tstep=n_tstep,
            dt=dt,
            f_ext_follower=f_ext,
            f_ext_dead=jnp.zeros_like(f_ext),
            f_ext_aero=None,
            prescribed_dofs=None,
        )

        # extract x coordinate
        x_t_out = 0.5 * (
            np.array(solution.hg[:, 0, 0, 3]) + np.array(solution.hg[:, 1, 0, 3])
        )

        def objective(
            ss: StructureFullStates, _: StructuralDesignVariables, i_ts: int
        ) -> Array:
            return jax.lax.select(
                i_ts == n_tstep - 1,
                0.5 * (ss.hg[0, 0, 3] + ss.hg[1, 0, 3]),
                jnp.zeros(()),
            )  # x coordinate of beam centroid

        # beam centroid should follow x = 0.5*(f/m)*t^2
        t = jnp.arange(n_tstep, dtype=float) * dt
        expected_x_t = 0.5 * f_mag / m_cs[0, 0, 0] * t * t + 0.5  # add inital offset

        grads, _ = beam.dynamic_adjoint(structure=solution, objective=objective)

        f_follower_grad = np.array(grads.f_ext_follower[:, 0, 0]).sum()
        m_cs_grad = np.array(grads.m_cs[0, 0, 0])

        assert jnp.allclose(expected_x_t, x_t_out), (
            "Primal solution time series does not match analytical solution"
        )

        expected_f_follower_grad = 0.5 / m_cs[0, 0, 0]
        expected_m_cs_grad = -0.5 * f_mag / m_cs[0, 0, 0] ** 2

        assert jnp.allclose(f_follower_grad, expected_f_follower_grad), (
            "Follower force gradient does not match analytical solution"
        )
        assert jnp.allclose(m_cs_grad, expected_m_cs_grad), (
            "Mass gradient does not match analytical solution"
        )
