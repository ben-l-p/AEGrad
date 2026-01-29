from aegrad.structure.structure import Structure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


class TestConstXVelocityXBeam:
    r"""
    Simulate a beam with constant initial rotational velocity about its midpoint
    """

    v_direction_index: int = 1
    beam_direction_index: int = 0
    y_vect = jnp.array([[0.0, 1.0, 0.0]])

    @classmethod
    def test_const_velocity_beam(cls):
        omega = 1.0
        l = 3.14
        coords = (
            jnp.zeros((3, 3))
            .at[:, cls.beam_direction_index]
            .set(jnp.array((-0.5 * l, 0.0, 0.5 * l)))
        )
        conn = jnp.array([[0, 1], [1, 2]])

        k_cs = jnp.diag(jnp.full(6, 1e6))
        m_bar = 5.0 * jnp.eye(3)
        j_bar = 0.1 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 500
        dt = 0.001

        struct = Structure(3, conn, cls.y_vect, None)
        struct.set_design_variables(coords, k_cs, m_cs)

        v_init = jnp.zeros((3, 6))
        v_init = v_init.at[:, cls.v_direction_index + 3].set(omega)
        if cls.beam_direction_index != cls.v_direction_index:
            v_init = v_init.at[0, :3].set(
                jnp.cross(
                    0.5 * l * jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                    jnp.zeros(3).at[cls.v_direction_index].set(omega),
                )
            )
            v_init = v_init.at[2, :3].set(-v_init[0, :3])

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init

        output = struct.dynamic_solve(
            init_cond,
            n_tstep,
            dt,
            None,
            None,
            None,
            max_iter=10,
            spectral_radius=0.9,
            abs_tol=1e-10,
        )

        node0x = np.array(output.hg[:, 0, :3, 3])
        node1x = np.array(output.hg[:, 1, :3, 3])
        node2x = np.array(output.hg[:, 2, :3, 3])

        node0v = np.array(output.v[:, 0, :])
        node1v = np.array(output.v[:, 1, :])
        node2v = np.array(output.v[:, 2, :])

        pass


TestConstXVelocityXBeam.test_const_velocity_beam()


# class TestConstYVelocityXBeam(TestConstXVelocityXBeam):
#     v_direction_index: int = 1
#
#
# class TestConstZVelocityXBeam(TestConstXVelocityXBeam):
#     v_direction_index: int = 2
#
#
# class TestConstXVelocityYBeam(TestConstXVelocityXBeam):
#     beam_direction_index: int = 1
#     y_vect = jnp.array([[0.0, 0.0, 1.0]])
#
#
# class TestConstYVelocityYBeam(TestConstXVelocityYBeam):
#     v_direction_index: int = 1
#
#
# class TestConstZVelocityYBeam(TestConstXVelocityYBeam):
#     v_direction_index: int = 2
#
#
# class TestConstXVelocityZBeam(TestConstXVelocityXBeam):
#     beam_direction_index: int = 2
#     y_vect = jnp.array([[1.0, 0.0, 0.0]])
#
#
# class TestConstYVelocityZBeam(TestConstXVelocityZBeam):
#     v_direction_index: int = 1
#
#
# class TestConstZVelocityZBeam(TestConstXVelocityZBeam):
#     v_direction_index: int = 2
