from jax.scipy.linalg import block_diag
from jax import numpy as jnp
import jax

from aegrad.structure import BeamStructure

jax.config.update("jax_enable_x64", True)


class TestXBeamConstXRotVelocity:
    r"""
    Test a two-node beam with lumped masses at each node, rotating with constant angular velocity about one of the
    principal axes (x_target, y, or z).
    """

    v_direction_index: int = 0
    beam_direction_index: int = 0
    y_vect = jnp.array([[0.0, 1.0, 0.0]])

    @classmethod
    def test_dynamic_rotating_beam(cls):
        n_tstep = 500
        dt = 0.001
        omega = 10.0
        r_ext = 4.56
        k_coeffs = jnp.full(6, 1e4)
        m_bar = 1.0
        j_bar = 1.0

        a = m_bar * r_ext * omega**2
        b = -3.0 * k_coeffs[0] - 2.0 * m_bar * r_ext**2 * omega**2
        c = m_bar * r_ext**3 * omega**2

        # solve to initialise beam with centrifugal deformation, such that there are no oscillations
        if cls.beam_direction_index != cls.v_direction_index:
            # solve a quadratic equation for deformation
            dx = jnp.roots(jnp.array([a, b, c]))[1].real
        else:
            dx = 0.0
        r_ref = r_ext - dx

        m_node = 0.5 * r_ref * m_bar

        coords = (
            jnp.zeros((3, 3))
            .at[:, cls.beam_direction_index]
            .set(jnp.array((-r_ref, 0.0, r_ref)))
        )
        conn = jnp.array([[0, 1], [1, 2]])

        m_cs = block_diag(jnp.eye(3) * m_bar, jnp.eye(3) * j_bar)[None, :]

        struct = BeamStructure(3, conn, cls.y_vect, None)
        struct.set_design_variables(coords, jnp.diag(k_coeffs), m_cs)

        v_init = jnp.zeros((3, 6))
        v_init = v_init.at[:, cls.v_direction_index + 3].set(omega)

        if cls.beam_direction_index != cls.v_direction_index:
            v_init = v_init.at[0, :3].set(
                jnp.cross(
                    r_ext * jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                    jnp.zeros(3).at[cls.v_direction_index].set(omega),
                )
            )
            v_init = v_init.at[2, :3].set(-v_init[0, :3])

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init

        if cls.beam_direction_index != cls.v_direction_index:
            init_cond.hg = init_cond.hg.at[0, cls.beam_direction_index, 3].set(-r_ext)
            init_cond.hg = init_cond.hg.at[2, cls.beam_direction_index, 3].set(r_ext)

        output = struct.dynamic_solve(
            init_state=init_cond,
            n_tstep=n_tstep,
            dt=dt,
            prescribed_dofs=None,
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
            spectral_radius=1.0,
            relaxation_factor=1.0,
        )

        expected_theta = omega * jnp.arange(n_tstep) * dt

        if cls.beam_direction_index != cls.v_direction_index:
            expected_f = m_node * r_ext * omega**2 * (2.0 / 3.0)
            x0_expected = jnp.zeros((n_tstep, 3))

            third_dir = (
                {0, 1, 2} - {cls.beam_direction_index, cls.v_direction_index}
            ).pop()
            third_dir_sign = jnp.sum(
                jnp.cross(
                    jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                    jnp.zeros(3).at[cls.v_direction_index].set(1.0),
                )
            )

            x0_expected = x0_expected.at[:, third_dir].set(
                r_ext * jnp.sin(expected_theta) * third_dir_sign
            )
            x0_expected = x0_expected.at[:, cls.beam_direction_index].set(
                -r_ext * jnp.cos(expected_theta)
            )

            x2_expected = -x0_expected
        else:
            expected_f = 0.0
            x0_expected = jnp.zeros(3).at[cls.beam_direction_index].set(-r_ref)
            x2_expected = jnp.zeros(3).at[cls.beam_direction_index].set(r_ref)
            third_dir = None
            third_dir_sign = None

        assert jnp.allclose(output.f_int[:, 0, cls.beam_direction_index], expected_f), (
            "Internal force does not match expected force at node 0"
        )
        assert jnp.allclose(
            output.f_int[:, 2, cls.beam_direction_index], -expected_f
        ), "Internal force does not match expected force at node 1"
        assert jnp.allclose(
            output.f_iner[:, 0, cls.beam_direction_index], -expected_f
        ), "Inertial force does not match expected force at node 0"
        assert jnp.allclose(
            output.f_iner[:, 2, cls.beam_direction_index], expected_f
        ), "Inertial force does not match expected force at node 1"
        assert jnp.allclose(output.hg[:, 0, :3, 3], x0_expected), (
            "Node 0 position does not match expected position"
        )
        assert jnp.allclose(output.hg[:, 2, :3, 3], x2_expected), (
            "Node 1 position does not match expected position"
        )

        assert jnp.allclose(output.v[:, :, cls.v_direction_index + 3], omega), (
            "Rotational velocity incorrect for all nodes."
        )
        assert jnp.allclose(output.v_dot, 0.0), "Accelerations are not zero."

        if cls.beam_direction_index != cls.v_direction_index:
            assert jnp.allclose(
                output.v[:, 0, third_dir], omega * r_ext * third_dir_sign
            ), "Linear velocity incorrect for node 0"
            assert jnp.allclose(
                output.v[:, 2, third_dir], -omega * r_ext * third_dir_sign
            ), "Linear velocity incorrect for node 1"
        else:
            assert jnp.allclose(output.v[:, 0, :3], jnp.zeros(3)), (
                "Linear velocity incorrect for node 0"
            )
            assert jnp.allclose(output.v[:, 2, :3], jnp.zeros(3)), (
                "Linear velocity incorrect for node 1"
            )


class TestXBeamConstYRotVelocity(TestXBeamConstXRotVelocity):
    v_direction_index: int = 1


class TestXBeamConstZRotVelocity(TestXBeamConstXRotVelocity):
    v_direction_index: int = 2


class TestYBeamConstXRotVelocity(TestXBeamConstXRotVelocity):
    beam_direction_index: int = 1
    y_vect = jnp.array([[0.0, 0.0, 1.0]])


class TestYBeamConstYRotVelocity(TestYBeamConstXRotVelocity):
    v_direction_index: int = 1


class TestYBeamConstZRotVelocity(TestYBeamConstXRotVelocity):
    v_direction_index: int = 2


class TestZBeamConstXRotVelocity(TestXBeamConstXRotVelocity):
    beam_direction_index: int = 2
    y_vect = jnp.array([[1.0, 0.0, 0.0]])


class TestZBeamConstYRotVelocity(TestZBeamConstXRotVelocity):
    v_direction_index: int = 1


class TestZBeamConstConstZRotVelocity(TestZBeamConstXRotVelocity):
    v_direction_index: int = 2
