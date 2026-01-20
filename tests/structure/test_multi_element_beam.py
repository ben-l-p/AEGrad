from jax import numpy as jnp
import jax
from aegrad.structure.structure import Structure

jax.config.update("jax_enable_x64", True)


class TestTwoNodeBeamStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    l = jnp.array(3.45)
    n_nodes = 8
    n_elem = n_nodes - 1
    coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(0, l, n_nodes))
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))
    y_vect = jnp.zeros((n_elem, 3)).at[:, 1].set(1.0)

    struct = Structure(n_nodes, conn, y_vect)

    @classmethod
    def test_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """

        k_coeffs_unloaded = jnp.full(6, 4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_unloaded)[None, :], None
        )
        d_unloaded = jnp.zeros((cls.n_elem, 6)).at[:, 0].set(cls.l / cls.n_elem)
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
        k_coeffs_axial = jnp.full(6, 1e5).at[0].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_axial)[None, :], None
        )
        dx = 0.1
        d_axial = jnp.zeros((cls.n_elem, 6))
        d_axial = d_axial.at[:, 0].set((cls.l + dx) / cls.n_elem)
        eps_axial = cls.struct.make_eps(d_axial)
        expected_axial_strain = jnp.array((dx / cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))[
            None, :
        ]
        expected_f_axial = k_coeffs_axial[0] * dx / cls.l

        assert jnp.allclose(eps_axial, expected_axial_strain), (
            f"Axial strain calculation incorrect, expected {expected_axial_strain}, got {eps_axial}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_axial)[0].reshape(cls.n_nodes, 6)

        assert jnp.allclose(g_int_axial[0, 0], -expected_f_axial), (
            f"Axial force calculation at root incorrect, expected {-expected_f_axial}, got {g_int_axial[0, 0]}"
        )

        assert jnp.allclose(g_int_axial[-1, 0], expected_f_axial), (
            f"Axial force calculation at tip incorrect, expected {expected_f_axial}, got {g_int_axial[-1, 0]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {0, 6 * (cls.n_nodes - 1)})
        )
        assert jnp.allclose(g_int_axial.ravel()[index_zero], 0.0), (
            f"Axial force in beam incorrect, expected zero, got {g_int_axial}"
        )

    @classmethod
    def test_torsional_strain(cls):
        r"""
        Ensure torsional strain and forces are calculated correctly.
        """
        k_coeffs_torsional = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_torsional)[None, :], None
        )
        dx = 0.1
        d_torsion = jnp.zeros((cls.n_elem, 6))
        d_torsion = d_torsion.at[:, 0].set(cls.l / cls.n_elem)
        d_torsion = d_torsion.at[:, 3].set(dx / cls.n_elem)

        eps_torsion = cls.struct.make_eps(d_torsion)
        expected_torsion_strain = jnp.array((0.0, 0.0, 0.0, dx / cls.l, 0.0, 0.0))[
            None, :
        ]
        expected_f_torsion = k_coeffs_torsional[3] * dx / cls.l

        assert jnp.allclose(eps_torsion, expected_torsion_strain), (
            f"Torsional strain calculation incorrect, expected {expected_torsion_strain}, got {eps_torsion}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_torsion)[0].reshape(
            cls.n_nodes, 6
        )

        assert jnp.allclose(g_int_axial[0, 3], -expected_f_torsion), (
            f"Torsional force calculation at root incorrect, expected {-expected_f_torsion}, got {g_int_axial[0, 3]}"
        )

        assert jnp.allclose(g_int_axial[-1, 3], expected_f_torsion), (
            f"Torsional force calculation at tip incorrect, expected {expected_f_torsion}, got {g_int_axial[-1, 3]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {3, 6 * (cls.n_nodes - 1) + 3})
        )
        assert jnp.allclose(g_int_axial.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {g_int_axial}"
        )

    @classmethod
    def test_bending_y_strain(cls):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """
        k_coeffs_bending = jnp.full(6, 1e5).at[4].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )
        dx = 0.1
        d_bending = jnp.zeros((cls.n_elem, 6))
        d_bending = d_bending.at[:, 0].set(cls.l / cls.n_elem)
        d_bending = d_bending.at[:, 4].set(dx / cls.n_elem)

        eps_bending = cls.struct.make_eps(d_bending)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, dx / cls.l, 0.0))[
            None, :
        ]
        expected_f_bending = k_coeffs_bending[4] * dx / cls.l

        assert jnp.allclose(eps_bending, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_bending)[0].reshape(
            cls.n_nodes, 6
        )

        assert jnp.allclose(g_int_axial[0, 4], -expected_f_bending), (
            f"Bending moment calculation at root incorrect, expected {-expected_f_bending}, got {g_int_axial[0, 4]}"
        )

        assert jnp.allclose(g_int_axial[-1, 4], expected_f_bending), (
            f"Bending calculation at tip incorrect, expected {expected_f_bending}, got {g_int_axial[-1, 4]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {4, 6 * (cls.n_nodes - 1) + 4})
        )
        assert jnp.allclose(g_int_axial.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {g_int_axial}"
        )

    @classmethod
    def test_bending_z_strain(cls):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """
        k_coeffs_bending = jnp.full(6, 1e5).at[5].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )
        dx = 0.1
        d_bending = jnp.zeros((cls.n_elem, 6))
        d_bending = d_bending.at[:, 0].set(cls.l / cls.n_elem)
        d_bending = d_bending.at[:, 5].set(dx / cls.n_elem)

        eps_bending = cls.struct.make_eps(d_bending)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, dx / cls.l))[
            None, :
        ]
        expected_f_bending = k_coeffs_bending[5] * dx / cls.l

        assert jnp.allclose(eps_bending, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_bending)[0].reshape(
            cls.n_nodes, 6
        )

        assert jnp.allclose(g_int_axial[0, 5], -expected_f_bending), (
            f"Bending moment calculation at root incorrect, expected {-expected_f_bending}, got {g_int_axial[0, 5]}"
        )

        assert jnp.allclose(g_int_axial[-1, 5], expected_f_bending), (
            f"Bending calculation at tip incorrect, expected {expected_f_bending}, got {g_int_axial[-1, 5]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {5, 6 * (cls.n_nodes - 1) + 5})
        )
        assert jnp.allclose(g_int_axial.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {g_int_axial}"
        )
