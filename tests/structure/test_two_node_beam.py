import jax
from jax import numpy as jnp
from jax import vmap, Array, jacobian
from aegrad.algebra.se3 import x_rmat_to_ha, exp_se3, hg_to_d, p
from aegrad.structure.structure import Structure


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
        assert jnp.allclose(eps_unloaded, 0.0), (
            f"Axial strain calculation incorrect, expected zero strain, got {eps_unloaded}"
        )
        g_int_unloaded = cls.struct.make_g_int_and_k_t(d_unloaded)[0]
        assert jnp.allclose(g_int_unloaded, 0.0), (
            f"Internal force vector incorrect, expected zero force, got {g_int_unloaded}"
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
        expected_axial_strain = jnp.array((dx / cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert jnp.allclose(eps_axial, expected_axial_strain), (
            f"Axial strain calculation incorrect, expected {expected_axial_strain}, got {eps_axial}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_axial)[0]
        expected_f_axial = jnp.zeros(12)
        expected_f_axial = expected_f_axial.at[0].set(-k_coeffs_axial[0] * dx / cls.l)
        expected_f_axial = expected_f_axial.at[6].set(k_coeffs_axial[0] * dx / cls.l)
        assert jnp.allclose(g_int_axial, expected_f_axial), (
            f"Axial force calculation incorrect, expected {expected_f_axial}, got {g_int_axial}"
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
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending_y}"
        )
        g_ind_bending_y = cls.struct.make_g_int_and_k_t(d_bending_y)[0]
        expected_f_bending_y = jnp.zeros(12)
        expected_f_bending_y = expected_f_bending_y.at[4].set(
            -k_coeffs_bending[4] * curvature_y
        )
        expected_f_bending_y = expected_f_bending_y.at[10].set(
            k_coeffs_bending[4] * curvature_y
        )
        assert jnp.allclose(g_ind_bending_y, expected_f_bending_y), (
            f"Bending moment calculation incorrect, expected {expected_f_bending_y}, got {g_ind_bending_y}"
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
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending_z}"
        )
        g_ind_bending_z = cls.struct.make_g_int_and_k_t(d_bending_z)[0]
        expected_f_bending_z = jnp.zeros(12)
        expected_f_bending_z = expected_f_bending_z.at[5].set(
            -k_coeffs_bending[5] * curvature_z
        )
        expected_f_bending_z = expected_f_bending_z.at[11].set(
            k_coeffs_bending[5] * curvature_z
        )
        assert jnp.allclose(g_ind_bending_z, expected_f_bending_z), (
            f"Bending moment calculation incorrect, expected {expected_f_bending_z}, got {g_ind_bending_z}"
        )

    @classmethod
    def test_undeformed_stiffness_matrix(cls):
        jax.config.update("jax_debug_nans", True)

        ha = vmap(lambda x_: x_rmat_to_ha(x_, jnp.eye(3)))(cls.coords)  # [2, 6]
        k_coeffs = jnp.linspace(1.0, 6.0, 6)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        d_base = jnp.zeros((1, 6)).at[0, 0].set(cls.l)
        k_t = cls.struct.make_g_int_and_k_t(d_base)[1]  # [12, 12]

        def make_force(ha_vect: Array) -> Array:
            hg0 = exp_se3(ha_vect[:6])
            hg1 = exp_se3(ha_vect[6:])
            d = hg_to_d(hg0, hg1)
            eps = (d - jnp.array([cls.l, 0.0, 0.0, 0.0, 0.0, 0.0])) / cls.l
            p_d = p(d)
            return p_d.T @ cls.struct.k[0, ...] @ eps

        k_t_ad = jacobian(make_force)(ha.ravel())

        pass
