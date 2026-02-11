from aegrad.structure.beam import BeamStructure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import jax
from jax import vmap
from aegrad.algebra.se3 import log_se3

jax.config.update("jax_enable_x64", True)


class TestConstLinXVelocityLumpedMass:
    r"""
    Simulate a point with constant initial rotational velocity.
    """

    v_direction_index: int = 0

    @classmethod
    def test_const_velocity_point_mass(cls):
        v = 10.0

        coords = jnp.zeros((1, 3))
        conn = jnp.zeros((0, 2), dtype=int)

        m = 0.1
        j = 10.0
        m_lump = block_diag(jnp.eye(3) * m, jnp.eye(3) * j)

        n_tstep = 50
        dt = 0.001

        struct = BeamStructure(1, conn, jnp.zeros((0, 3)), None)
        struct.set_design_variables(
            coords, jnp.zeros((0, 6, 6)), None, m_lump[None, ...]
        )

        v_init = jnp.zeros((1, 6)).at[0, cls.v_direction_index].set(v)

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init

        output = struct.dynamic_solve(
            init_cond,
            n_tstep,
            dt,
            None,
            None,
            None,
            max_n_iter=10,
            spectral_radius=1.0,
        )

        x_expected = v * jnp.arange(n_tstep) * dt

        disp_measured = output.hg[:, 0, :3, 3]
        theta_measured = vmap(log_se3)(output.hg[:, 0, :, :])[:, 3:]
        x_measured = jnp.concatenate((disp_measured, theta_measured), axis=-1)[
            :, cls.v_direction_index
        ]

        assert jnp.allclose(x_measured, x_expected), (
            "Displacements/angles do not match expected values."
        )

        assert jnp.allclose(output.v[:, 0, cls.v_direction_index], v), (
            "Velocities do not remain constant as expected."
        )

        assert jnp.allclose(output.v_dot, 0.0), (
            "Accelerations are not zero as expected."
        )


class TestConstLinYVelocityLumpedMass(TestConstLinXVelocityLumpedMass):
    v_direction_index: int = 1


class TestConstLinZVelocityLumpedMass(TestConstLinXVelocityLumpedMass):
    v_direction_index: int = 2


class TestConstRotXVelocityLumpedMass(TestConstLinXVelocityLumpedMass):
    v_direction_index: int = 3


class TestConstRotYVelocityLumpedMass(TestConstLinXVelocityLumpedMass):
    v_direction_index: int = 4


class TestConstRotZVelocityLumpedMass(TestConstLinXVelocityLumpedMass):
    v_direction_index: int = 5
