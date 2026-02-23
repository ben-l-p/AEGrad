from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import jax

from aegrad.structure import BeamStructure

jax.config.update("jax_enable_x64", True)


class TestXGravityXBeamDrop:
    g_direction_index: int = 0
    beam_direction_index: int = 0
    y_vect = jnp.array([[0.0, 1.0, 0.0]])

    @classmethod
    def test_beam_drop(cls):
        length = 3.14
        g = -9.81

        g_vec = jnp.zeros(3).at[cls.g_direction_index].set(g)

        coords = jnp.zeros((2, 3)).at[1, cls.beam_direction_index].set(length)
        conn = jnp.array([[0, 1]])

        k_cs = jnp.diag(jnp.full(6, 1e3))
        m_bar = 5.0 * jnp.eye(3)
        j_bar = 0.1 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 1000
        dt = 0.001

        struct = BeamStructure(2, conn, cls.y_vect, g_vec)
        struct.set_design_variables(coords, k_cs, m_cs)

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v_dot = init_cond.v_dot.at[:, :3].set(g_vec[None, :])

        output = struct.dynamic_solve(
            init_state=init_cond,
            n_tstep=n_tstep,
            dt=dt,
            prescribed_dofs=None,
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
            spectral_radius=1.0,
        )

        expected_nodal_fg = 0.5 * m_bar[0, 0] * length * g
        expected_v = jnp.arange(1, n_tstep) * dt * g
        expected_x0 = 0.5 * g * (jnp.arange(1, n_tstep) * dt) ** 2
        expected_x1 = expected_x0 + coords[1, cls.g_direction_index]

        assert jnp.allclose(
            expected_x0,
            output.hg[1:, 0, cls.g_direction_index, 3],
        ), "Node 0 positions do not match expected values"
        assert jnp.allclose(
            expected_x1,
            output.hg[1:, 1, cls.g_direction_index, 3],
        ), "Node 1 positions do not match expected values"
        assert jnp.allclose(expected_v, output.v[1:, 0, cls.g_direction_index]), (
            "Node 0 velocities do not match expected values"
        )
        assert jnp.allclose(expected_v, output.v[1:, 1, cls.g_direction_index]), (
            "Node 1 velocities do not match expected values"
        )
        assert jnp.allclose(g, output.v_dot[:, :, cls.g_direction_index]), (
            "Accelerations do not match gravity"
        )
        assert jnp.allclose(
            output.f_grav[1:, :, cls.g_direction_index], expected_nodal_fg
        ), "Gravitational forces do not match expected values"


class TestYGravityXBeamDrop(TestXGravityXBeamDrop):
    g_direction_index: int = 1


class TestZGravityXBeamDrop(TestXGravityXBeamDrop):
    g_direction_index: int = 2


class TestXGravityYBeamDrop(TestXGravityXBeamDrop):
    beam_direction_index: int = 1
    y_vect = jnp.array([[0.0, 0.0, 1.0]])


class TestYGravityYBeamDrop(TestXGravityYBeamDrop):
    g_direction_index: int = 1


class TestZGravityYBeamDrop(TestXGravityYBeamDrop):
    g_direction_index: int = 2


class TestXGravityZBeamDrop(TestXGravityXBeamDrop):
    beam_direction_index: int = 2
    y_vect = jnp.array([[1.0, 0.0, 0.0]])


class TestYGravityZBeamDrop(TestXGravityZBeamDrop):
    g_direction_index: int = 1


class TestZGravityZBeamDrop(TestXGravityZBeamDrop):
    g_direction_index: int = 2
