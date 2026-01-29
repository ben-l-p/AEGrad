from aegrad.structure.structure import Structure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)


class TestConstXVelocityXBeam:
    v_direction_index: int = 0
    beam_direction_index: int = 0
    y_vect = jnp.array([[0.0, 1.0, 0.0]])

    @classmethod
    def test_const_velocity_beam(cls):
        v_mag: float = 50.0
        l = 3.14

        coords = jnp.zeros((2, 3)).at[1, cls.beam_direction_index].set(l)
        conn = jnp.array([[0, 1]])

        k_cs = jnp.diag(jnp.full(6, 1e9))
        m_bar = 5.0 * jnp.eye(3)
        j_bar = 0.1 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 5
        dt = 0.001

        struct = Structure(2, conn, cls.y_vect, None)
        struct.set_design_variables(coords, k_cs, m_cs)

        v_init = jnp.zeros((2, 6)).at[:, cls.v_direction_index].set(v_mag)

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init

        output = struct.dynamic_solve(
            init_cond, n_tstep, dt, None, None, None, max_iter=5
        )
        x_t = np.array(output.hg[:, 0, cls.v_direction_index, 3])  # [n_tstep]

        expected_x_t = jnp.arange(n_tstep) * dt * v_mag  # [n_tstep]

        assert np.allclose(expected_x_t, x_t), (
            "Beam with constant initial velocity did not maintain constant velocity."
        )


class TestConstYVelocityXBeam(TestConstXVelocityXBeam):
    v_direction_index: int = 1


class TestConstZVelocityXBeam(TestConstXVelocityXBeam):
    v_direction_index: int = 2


class TestConstXVelocityYBeam(TestConstXVelocityXBeam):
    beam_direction_index: int = 1
    y_vect = jnp.array([[0.0, 0.0, 1.0]])


class TestConstYVelocityYBeam(TestConstXVelocityYBeam):
    v_direction_index: int = 1


class TestConstZVelocityYBeam(TestConstXVelocityYBeam):
    v_direction_index: int = 2


class TestConstXVelocityZBeam(TestConstXVelocityXBeam):
    beam_direction_index: int = 2
    y_vect = jnp.array([[1.0, 0.0, 0.0]])


class TestConstYVelocityZBeam(TestConstXVelocityZBeam):
    v_direction_index: int = 1


class TestConstZVelocityZBeam(TestConstXVelocityZBeam):
    v_direction_index: int = 2
