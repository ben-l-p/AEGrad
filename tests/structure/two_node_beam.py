from aegrad.structure.structure import Structure
from jax import numpy as jnp


class TestTwoNodeBeamStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    l = 3.14
    coords = jnp.zeros((2, 3)).at[1, 0].set(l)
    struct = Structure(2, jnp.array([[0, 1]]), jnp.array([[0.0, 1.0, 0.0]]))

    @classmethod
    def test_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """

        k_coeffs_unloaded = jnp.ones(6)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_unloaded)[None, :], None
        )
        d_unloaded = jnp.zeros((1, 6)).at[0, 0].set(cls.l)
        eps_unloaded = cls.struct.make_eps(d_unloaded)
        assert jnp.allclose(eps_unloaded, 0.0), "Axial strain calculation incorrect"
        g_int_unloaded, _ = cls.struct.make_g_int_and_k_t(d_unloaded)
        assert jnp.allclose(g_int_unloaded, 0.0), (
            "Internal force vector should be zero for unloaded structure"
        )

    @classmethod
    def test_axial_strain(cls):
        r"""
        Ensure axial strain and forces are calculated correctly.
        """
        k_coeffs_axial = jnp.full(6, 1e5).at[0].set(cls.l)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_axial)[None, :], None
        )
        dx = 0.1
        d_axial = jnp.zeros((1, 6))
        d_axial = d_axial.at[0, 0].set(cls.l + dx)
        eps_axial = cls.struct.make_eps(d_axial)
        assert jnp.allclose(
            eps_axial, jnp.array((dx / cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))
        ), "Axial strain calculation incorrect"
        g_int_axial, _ = cls.struct.make_g_int_and_k_t(d_axial)
        expected_f_axial = jnp.zeros(12)
        expected_f_axial = expected_f_axial.at[0].set(-k_coeffs_axial[0] * dx / cls.l)
        expected_f_axial = expected_f_axial.at[6].set(k_coeffs_axial[0] * dx / cls.l)
        assert jnp.allclose(g_int_axial, expected_f_axial), (
            "Axial force calculation incorrect"
        )

    @classmethod
    def test_bending_y(cls):
        r"""
        Ensure bending in y strain and moments are calculated correctly.
        """
        k_coeffs_bending = jnp.full(6, 3e5).at[4].set(1.0)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )
        curvature_y = 0.1
        d_bending_y = jnp.zeros((1, 6))
        d_bending_y = d_bending_y.at[0, 0].set(cls.l)
        d_bending_y = d_bending_y.at[0, 4].set(curvature_y * cls.l)
        eps_bending_y = cls.struct.make_eps(d_bending_y)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, curvature_y, 0.0))
        assert jnp.allclose(eps_bending_y, expected_bending_strain), (
            "Bending strain calculation incorrect"
        )
        g_ind_bending_y, _ = cls.struct.make_g_int_and_k_t(d_bending_y)
        expected_f_bending_y = jnp.zeros(12)
        expected_f_bending_y = expected_f_bending_y.at[4].set(
            -k_coeffs_bending[4] * curvature_y
        )
        expected_f_bending_y = expected_f_bending_y.at[10].set(
            k_coeffs_bending[4] * curvature_y
        )
        assert jnp.allclose(g_ind_bending_y, expected_f_bending_y), (
            "Bending moment calculation incorrect"
        )

    @classmethod
    def test_bending_z(cls):
        r"""
        Ensure bending in z strain and moments are calculated correctly.
        """
        k_coeffs_bending = jnp.full(6, 3e5).at[5].set(1.0)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )
        curvature_z = 0.1
        d_bending_z = jnp.zeros((1, 6))
        d_bending_z = d_bending_z.at[0, 0].set(cls.l)
        d_bending_z = d_bending_z.at[0, 5].set(curvature_z * cls.l)
        eps_bending_z = cls.struct.make_eps(d_bending_z)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, curvature_z))
        assert jnp.allclose(eps_bending_z, expected_bending_strain), (
            "Bending strain calculation incorrect"
        )
        g_ind_bending_z, _ = cls.struct.make_g_int_and_k_t(d_bending_z)
        expected_f_bending_z = jnp.zeros(12)
        expected_f_bending_z = expected_f_bending_z.at[5].set(
            -k_coeffs_bending[5] * curvature_z
        )
        expected_f_bending_z = expected_f_bending_z.at[11].set(
            k_coeffs_bending[5] * curvature_z
        )
        assert jnp.allclose(g_ind_bending_z, expected_f_bending_z), (
            "Bending moment calculation incorrect"
        )
