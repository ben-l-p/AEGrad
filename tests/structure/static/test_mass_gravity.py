from jax import numpy as jnp
import jax

from aegrad.structure import BeamStructure
from aegrad.algebra.se3 import exp_se3

jax.config.update("jax_enable_x64", True)


class TestTwoNodeXGravityZ:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    beam_direction = "x_target"
    direction_index = 0
    y_vector = jnp.array([[0.0, 1.0, 0.0]])

    g_vec = jnp.array([0.0, 0.0, -9.81])  # gravity vector

    l = jnp.array(2.5)
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(l)

    m_bar = 10.0  # mass per unit b_ref

    k_cs = jnp.eye(6) * 1e6  # stiffness matrix
    m_cs = jnp.zeros((6, 6)).at[:3, :3].set(m_bar * jnp.eye(3))

    @classmethod
    def test_total_mass(cls):
        r"""
        Ensure total mass of beam is correct
        """

        cls.struct = BeamStructure(2, jnp.array([[0, 1]]), cls.y_vector, cls.g_vec)
        cls.struct.set_design_variables(cls.coords, cls.k_cs, cls.m_cs)

        m_t = cls.struct._make_m_t(cls.struct.d0)

        expected_mass = cls.m_bar * cls.l

        # consider all three directions, and divide by three
        matrix_mass = jnp.sum(
            m_t
            @ jnp.array((1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
            / 3.0
        )

        assert jnp.allclose(matrix_mass, expected_mass), (
            f"Total mass from mass matrix {matrix_mass} does not match expected {expected_mass}"
        )

    @classmethod
    def test_gravity_forces(cls):
        r"""
        Ensure weight of beam is correct
        """

        cls.struct = BeamStructure(2, jnp.array([[0, 1]]), cls.y_vector, cls.g_vec)
        cls.struct.set_design_variables(cls.coords, cls.k_cs, cls.m_cs)
        f_g = cls.struct._assemble_vector_from_entries(
            cls.struct._make_f_grav(
                cls.struct._make_m_t(cls.struct.d0), cls.struct.hg0[:, :3, :3]
            )
        )

        expected_weight = cls.m_bar * cls.l * cls.g_vec
        matrix_weight = (
            cls.struct.hg0[0, :3, :3] @ f_g[:3] + cls.struct.hg0[1, :3, :3] @ f_g[6:9]
        )

        assert jnp.allclose(matrix_weight, expected_weight), (
            f"""Weight from gravity forces {matrix_weight} does not match expected {expected_weight}"""
        )

    @classmethod
    def test_gravity_forces_deformed(cls):
        r"""
        Ensure weight of beam is correct
        """

        cls.struct = BeamStructure(2, jnp.array([[0, 1]]), cls.y_vector, cls.g_vec)
        cls.struct.set_design_variables(cls.coords, cls.k_cs, cls.m_cs)

        # make beam curved around local y
        d = jnp.array((cls.l, 0.0, 0.0, 0.0, jnp.pi / 2.0, 0.0))
        ha0 = jnp.eye(4).at[:3, :3].set(cls.struct.o0[0, ...])
        hb = ha0 @ exp_se3(d) @ ha0.T

        hg = jnp.stack((jnp.eye(4), hb), axis=0)

        f_g = cls.struct._assemble_vector_from_entries(
            cls.struct._make_f_grav(cls.struct._make_m_t(d[None, :]), hg[:, :3, :3])
        )

        expected_weight = cls.m_bar * cls.l * cls.g_vec
        matrix_weight = hg[0, :3, :3] @ f_g[:3] + hg[1, :3, :3] @ f_g[6:9]

        assert jnp.allclose(matrix_weight, expected_weight), (
            f"""Weight from gravity forces {matrix_weight} does not match expected {expected_weight}"""
        )


class TestTwoNodeXGravityX(TestTwoNodeXGravityZ):
    g_vec = jnp.array([-9.81, 0.0, 0.0])


class TestTwoNodeXGravityY(TestTwoNodeXGravityZ):
    g_vec = jnp.array([0.0, -9.81, 0.0])


class TestTwoNodeYGravityZ(TestTwoNodeXGravityZ):
    beam_direction = "y"
    direction_index = 1
    y_vector = jnp.array([[0.0, 0.0, 1.0]])

    coords = jnp.zeros((2, 3)).at[1, direction_index].set(TestTwoNodeXGravityZ.l)


class TestTwoNodeYGravityX(TestTwoNodeYGravityZ):
    g_vec = jnp.array([-9.81, 0.0, 0.0])


class TestTwoNodeYGravityY(TestTwoNodeYGravityZ):
    g_vec = jnp.array([0.0, -9.81, 0.0])


class TestTwoNodeZGravityZ(TestTwoNodeXGravityZ):
    beam_direction = "z"
    direction_index = 2
    y_vector = jnp.array([[1.0, 0.0, 0.0]])

    coords = jnp.zeros((2, 3)).at[1, direction_index].set(TestTwoNodeXGravityZ.l)


class TestTwoNodeZGravityX(TestTwoNodeZGravityZ):
    g_vec = jnp.array([-9.81, 0.0, 0.0])


class TestTwoNodeZGravityY(TestTwoNodeZGravityZ):
    g_vec = jnp.array([0.0, -9.81, 0.0])
