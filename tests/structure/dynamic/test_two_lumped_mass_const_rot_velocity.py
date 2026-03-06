from jax.scipy.linalg import block_diag
from jax import numpy as jnp
import jax

from aegrad.structure import BeamStructure

jax.config.update("jax_enable_x64", True)


class TestTwoXLumpedMassConstXRotVelocity:
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
        r_ext = 3.14
        k_coeffs = jnp.full(6, 1e4)
        m_lumped = 1.0
        j_lumped = 1.0

        # solve to initialise beam with centrifugal deformation, such that there are no oscillations
        if cls.beam_direction_index != cls.v_direction_index:
            dx = (
                m_lumped
                * r_ext**2
                * omega**2
                / (k_coeffs[0] + m_lumped * r_ext * omega**2)
            )
        else:
            dx = 0.0
        r_ref = r_ext - dx

        coords = (
            jnp.zeros((2, 3))
            .at[:, cls.beam_direction_index]
            .set(jnp.array((-r_ref, r_ref)))
        )
        conn = jnp.array([[0, 1]])

        m_mat_lumped = block_diag(jnp.eye(3) * m_lumped, jnp.eye(3) * j_lumped)[None, :]

        struct = BeamStructure(2, conn, cls.y_vect, None)
        struct.set_design_variables(
            coords, jnp.diag(k_coeffs), None, m_lumped=m_mat_lumped
        )

        v_init = jnp.zeros((2, 6))
        v_init = v_init.at[:, cls.v_direction_index + 3].set(omega)
        v_dot_init = jnp.zeros((2, 6))

        if cls.beam_direction_index != cls.v_direction_index:
            v_init = v_init.at[0, :3].set(
                jnp.cross(
                    r_ext * jnp.zeros(3).at[cls.beam_direction_index].set(1.0),
                    jnp.zeros(3).at[cls.v_direction_index].set(omega),
                )
            )
            v_init = v_init.at[1, :3].set(-v_init[0, :3])

        init_cond = struct.reference_configuration().to_dynamic()
        init_cond.v = v_init
        init_cond.v_dot = v_dot_init

        if cls.beam_direction_index != cls.v_direction_index:
            init_cond.hg = init_cond.hg.at[0, cls.beam_direction_index, 3].set(-r_ext)
            init_cond.hg = init_cond.hg.at[1, cls.beam_direction_index, 3].set(r_ext)

        output = struct.dynamic_solve(
            init_state=init_cond,
            n_tstep=n_tstep,
            dt=dt,
            prescribed_dofs=None,
            f_ext_aero=None,
            f_ext_dead=None,
            f_ext_follower=None,
            spectral_radius=1.0,
            relaxation_factor=1.0,
        )

        expected_theta = omega * jnp.arange(n_tstep) * dt

        if cls.beam_direction_index != cls.v_direction_index:
            expected_f = m_lumped * r_ext * omega**2
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
            output.f_int[:, 1, cls.beam_direction_index], -expected_f
        ), "Internal force does not match expected force at node 1"
        assert jnp.allclose(
            output.f_iner[:, 0, cls.beam_direction_index], -expected_f
        ), "Inertial force does not match expected force at node 0"
        assert jnp.allclose(
            output.f_iner[:, 1, cls.beam_direction_index], expected_f
        ), "Inertial force does not match expected force at node 1"
        assert jnp.allclose(output.hg[:, 0, :3, 3], x0_expected), (
            "Node 0 position does not match expected position"
        )
        assert jnp.allclose(output.hg[:, 1, :3, 3], x2_expected), (
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
                output.v[:, 1, third_dir], -omega * r_ext * third_dir_sign
            ), "Linear velocity incorrect for node 1"
        else:
            assert jnp.allclose(output.v[:, 0, :3], jnp.zeros(3)), (
                "Linear velocity incorrect for node 0"
            )
            assert jnp.allclose(output.v[:, 1, :3], jnp.zeros(3)), (
                "Linear velocity incorrect for node 1"
            )


class TestTwoXLumpedMassConstYRotVelocity(TestTwoXLumpedMassConstXRotVelocity):
    v_direction_index: int = 1


class TestTwoXLumpedMassConstZRotVelocity(TestTwoXLumpedMassConstXRotVelocity):
    v_direction_index: int = 2


class TestTwoYLumpedMassConstXRotVelocity(TestTwoXLumpedMassConstXRotVelocity):
    beam_direction_index: int = 1
    y_vect = jnp.array([[0.0, 0.0, 1.0]])


class TestTwoYLumpedMassConstYRotVelocity(TestTwoYLumpedMassConstXRotVelocity):
    v_direction_index: int = 1


class TestTwoYLumpedMassConstZRotVelocity(TestTwoYLumpedMassConstXRotVelocity):
    v_direction_index: int = 2


class TestTwoZLumpedMassConstXRotVelocity(TestTwoXLumpedMassConstXRotVelocity):
    beam_direction_index: int = 2
    y_vect = jnp.array([[1.0, 0.0, 0.0]])


class TestTwoZLumpedMassConstYRotVelocity(TestTwoZLumpedMassConstXRotVelocity):
    v_direction_index: int = 1


class TestTwoZLumpedMassConstZRotVelocity(TestTwoZLumpedMassConstXRotVelocity):
    v_direction_index: int = 2
