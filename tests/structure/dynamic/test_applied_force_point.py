from aegrad.structure.structure import Structure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import jax
from jax import vmap
from aegrad.algebra.se3 import log_se3

jax.config.update("jax_enable_x64", True)


class TestLinXForcePoint:
    r"""
    Simulate a point being acted on by a force.
    """

    f_direction_index: int = 0
    f: float = 34.15

    @classmethod
    def test_force_point_mass(cls):
        coords = jnp.zeros((1, 3))
        conn = jnp.zeros((0, 2), dtype=int)

        m = 0.1
        j = 10.0
        m_lump = block_diag(jnp.eye(3) * m, jnp.eye(3) * j)

        n_tstep = 50
        dt = 0.001

        struct = Structure(
            1,
            conn,
            jnp.zeros((0, 3)),
        )

        struct.set_design_variables(
            coords, jnp.zeros((0, 6, 6)), None, m_lump[None, ...]
        )

        v_dot_expected = cls.f / m_lump[cls.f_direction_index, cls.f_direction_index]

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v_dot = init_cond.v_dot.at[0, cls.f_direction_index].set(
            v_dot_expected
        )
        init_cond.a = init_cond.a.at[0, cls.f_direction_index].set(v_dot_expected)

        output = struct.dynamic_solve(
            init_cond,
            n_tstep,
            dt,
            jnp.zeros((1, 6)).at[0, cls.f_direction_index].set(cls.f),
            None,
            None,
            max_iter=10,
            spectral_radius=1.0,
            abs_tol=1e-10,
            relaxation_factor=1.0,
        )

        v_expected = jnp.arange(1, n_tstep) * dt * v_dot_expected
        x_expected = 0.5 * v_dot_expected * (jnp.arange(1, n_tstep) * dt) ** 2

        disp_measured = output.hg[:, 0, :3, 3]
        theta_measured = vmap(log_se3)(output.hg[:, 0, :, :])[:, 3:]
        x_measured = jnp.concatenate((disp_measured, theta_measured), axis=-1)[
            1:, cls.f_direction_index
        ]

        v_measured = output.v[1:, 0, cls.f_direction_index]
        v_dot_measured = output.v_dot[1:, 0, cls.f_direction_index]

        assert jnp.allclose(x_expected, x_measured), (
            "Displacements do not match expected values"
        )
        assert jnp.allclose(v_expected, v_measured), (
            "Velocities do not match expected values"
        )
        assert jnp.allclose(v_dot_expected, v_dot_measured), (
            "Accelerations do not match expected values"
        )


class TestLinYForcePoint(TestLinXForcePoint):
    f_direction_index: int = 1


class TestLinZForcePoint(TestLinXForcePoint):
    f_direction_index: int = 2


class TestRotXForcePoint(TestLinXForcePoint):
    f_direction_index: int = 3


class TestRotYForcePoint(TestLinXForcePoint):
    f_direction_index: int = 4


class TestRotZForcePoint(TestLinXForcePoint):
    f_direction_index: int = 5
