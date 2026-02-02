from aegrad.structure.structure import Structure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import jax

jax.config.update("jax_enable_x64", True)


class TestConstXVelocityXBeam:
    r"""
    Simulate a beam with constant initial rotational velocity about its midpoint
    """

    v_direction_index: int = 0
    beam_direction_index: int = 0
    y_vect = jnp.array([[0.0, 1.0, 0.0]])

    @classmethod
    def test_const_velocity_beam(cls):
        omega = 10.0
        l = 3.14

        coords = (
            jnp.zeros((3, 3))
            .at[:, cls.beam_direction_index]
            .set(jnp.array((-0.5 * l, 0.0, 0.5 * l)))
        )
        conn = jnp.array([[0, 1], [1, 2]])

        k_cs = jnp.diag(jnp.full(6, 1e9))
        m_bar = 0.1 * jnp.eye(3)
        j_bar = 100.0 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 500
        dt = 0.005

        struct = Structure(3, conn, cls.y_vect, None)
        struct.set_design_variables(coords, k_cs, m_cs)

        v_init = jnp.zeros((3, 6))
        v_init = v_init.at[:, cls.v_direction_index + 3].set(omega)
        v_dot_init = jnp.zeros((3, 6))
        if cls.beam_direction_index != cls.v_direction_index:
            v_init = v_init.at[0, :3].set(
                jnp.cross(
                    0.5 * l * jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                    jnp.zeros(3).at[cls.v_direction_index].set(omega),
                )
            )
            v_init = v_init.at[2, :3].set(-v_init[0, :3])

            # v_dot_init = v_dot_init.at[0, 0].set(0.5 * l * omega * omega)
            # v_dot_init = v_dot_init.at[2, 0].set(-0.5 * l * omega * omega)

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init
        init_cond.v_dot = v_dot_init

        output = struct.dynamic_solve(
            init_cond,
            n_tstep,
            dt,
            None,
            None,
            None,
            max_iter=10,
            spectral_radius=1.0,
            abs_tol=1e-10,
            relaxation_factor=1.0,
        )

        theta_expected = jnp.arange(n_tstep) * omega * dt

        if cls.beam_direction_index != cls.v_direction_index:
            x0_expected = jnp.zeros((n_tstep, 3))

            third_dir = (
                {0, 1, 2} - {cls.beam_direction_index, cls.v_direction_index}
            ).pop()
            x0_expected = x0_expected.at[:, third_dir].set(
                0.5
                * l
                * jnp.sin(theta_expected)
                * jnp.sum(
                    jnp.cross(
                        jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                        jnp.zeros(3).at[cls.v_direction_index].set(1.0),
                    )
                )
            )
            x0_expected = x0_expected.at[:, cls.beam_direction_index].set(
                -0.5 * l * jnp.cos(theta_expected)
            )

            x2_expected = -x0_expected
        else:
            x0_expected = jnp.zeros(3).at[cls.beam_direction_index].set(-0.5 * l)
            x2_expected = jnp.zeros(3).at[cls.beam_direction_index].set(0.5 * l)

        import numpy as np

        v0 = np.array(output.v[:, 0, :])
        v1 = np.array(output.v[:, 1, :])
        v2 = np.array(output.v[:, 2, :])
        v_dot0 = np.array(output.v_dot[:, 0, :])
        v_dot1 = np.array(output.v_dot[:, 1, :])
        v_dot2 = np.array(output.v_dot[:, 2, :])
        x0 = np.array(output.hg[:, 0, :3, 3])
        x1 = np.array(output.hg[:, 1, :3, 3])
        x2 = np.array(output.hg[:, 2, :3, 3])
        eps0 = np.array(output.eps[:, 0, :])
        eps1 = np.array(output.eps[:, 1, :])

        # output.plot(Path("./test_outputs/rotating_beam/"))

        x0 = output.hg[:, 0, :3, 3]
        x1 = output.hg[:, 1, :3, 3]
        x2 = output.hg[:, 2, :3, 3]

        v0 = output.v[:, 0, :]
        v1 = output.v[:, 1, :]
        v2 = output.v[:, 2, :]

        assert jnp.allclose(x0, x0_expected, atol=5e-4), (
            "Node 0 position does not match expected values"
        )
        assert jnp.allclose(x1, 0.0, atol=5e-4), (
            "Node 1 position does not match expected values"
        )
        assert jnp.allclose(x2, x2_expected, atol=5e-4), (
            "Node 2 position does not match expected values"
        )
        assert jnp.allclose(v0, v_init[None, 0, :], atol=1e-4), (
            "Node 0 velocity does not match expected values"
        )
        assert jnp.allclose(v1, v_init[None, 1, :], atol=1e-4), (
            "Node 1 velocity does not match expected values"
        )
        assert jnp.allclose(v2, v_init[None, 2, :], atol=1e-4), (
            "Node 2 velocity does not match expected values"
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
