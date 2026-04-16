from typing import cast

from jax import numpy as jnp
from jax.scipy.linalg import block_diag
from jax import Array

from aegrad.structure.beam import BaseBeamStructure


class TestXGravityPointDrop:
    r"""
    Simulate a point falling under gravity in the X direction.
    """

    g_direction_index: int = 0
    g: float = -9.81

    @classmethod
    def test_gravity_point_mass(cls):
        coords = jnp.zeros((1, 3))
        conn = jnp.zeros((0, 2), dtype=int)

        m = 0.1
        j = 10.0
        m_lump = block_diag(jnp.eye(3) * m, jnp.eye(3) * j)

        n_tstep = 50
        dt = 0.001

        struct = BaseBeamStructure(
            num_nodes=1,
            connectivity=conn,
            y_vector=jnp.zeros((0, 3)),
            gravity=jnp.zeros(3).at[cls.g_direction_index].set(cls.g),
            m_lumped_index=jnp.zeros((1,), dtype=int),
        )
        struct.set_design_variables(
            coords, jnp.zeros((0, 6, 6)), None, m_lump[None, ...]
        )

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v_dot = init_cond.v_dot.at[:, cls.g_direction_index].set(cls.g)

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

        expected_fg = jnp.zeros(6).at[cls.g_direction_index].set(m * cls.g)
        expected_v = jnp.arange(1, n_tstep) * dt * cls.g
        expected_x = 0.5 * cls.g * (jnp.arange(1, n_tstep) * dt) ** 2

        output_x = output.hg[1:, 0, cls.g_direction_index, 3]
        output_v = output.v[1:, 0, cls.g_direction_index]
        output_f = cast(Array, output.f_grav)[1:, 0, :]

        assert jnp.allclose(
            expected_x,
            output_x,
        ), "Node positions do not match expected values"

        assert jnp.allclose(expected_v, output_v), (
            "Node velocities do not match expected values"
        )

        assert jnp.allclose(expected_fg[None, :], output_f), (
            "Node force do not match expected value"
        )


class TestYGravityPointDrop(TestXGravityPointDrop):
    g_direction_index: int = 1


class TestZGravityPointDrop(TestXGravityPointDrop):
    g_direction_index: int = 2
