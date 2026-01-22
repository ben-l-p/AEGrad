from jax import numpy as jnp
from jax import vmap
import jax
from aegrad.structure.structure import Structure
from aegrad.algebra.test_routines import const_curvature_beam
from algebra.se3 import exp_se3

jax.config.update("jax_enable_x64", True)


class TestTwoNodeBeamStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    l = jnp.array(3.45)
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
        g_int_unloaded = cls.struct.make_g_int_and_k_t(d_unloaded, True, True)[0]
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
        d_axial = jnp.zeros((1, 6))
        d_axial = d_axial.at[0, 0].set(cls.l + dx)
        eps_axial = cls.struct.make_eps(d_axial)
        expected_axial_strain = jnp.array((dx / cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert jnp.allclose(eps_axial, expected_axial_strain), (
            f"Axial strain calculation incorrect, expected {expected_axial_strain}, got {eps_axial}"
        )
        g_int_axial = cls.struct.make_g_int_and_k_t(d_axial, True, True)[0]
        expected_f_axial = jnp.zeros(12)
        expected_f_axial = expected_f_axial.at[0].set(-k_coeffs_axial[0] * dx / cls.l)
        expected_f_axial = expected_f_axial.at[6].set(k_coeffs_axial[0] * dx / cls.l)
        assert jnp.allclose(g_int_axial, expected_f_axial), (
            f"Axial force calculation incorrect, expected {expected_f_axial}, got {g_int_axial}"
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
        theta_x = 0.1
        d_torsional = jnp.zeros((1, 6))
        d_torsional = d_torsional.at[0, 0].set(cls.l)
        d_torsional = d_torsional.at[0, 3].set(theta_x)
        eps_torsional = cls.struct.make_eps(d_torsional)
        expected_torsional_strain = jnp.array(
            (0.0, 0.0, 0.0, theta_x / cls.l, 0.0, 0.0)
        )
        assert jnp.allclose(eps_torsional, expected_torsional_strain), (
            f"Torsional strain calculation incorrect, expected {expected_torsional_strain}, got {eps_torsional}"
        )
        g_int_torsional = cls.struct.make_g_int_and_k_t(d_torsional, True, True)[0]
        expected_f_torsional = jnp.zeros(12)
        expected_f_torsional = expected_f_torsional.at[3].set(
            -k_coeffs_torsional[3] * theta_x / cls.l
        )
        expected_f_torsional = expected_f_torsional.at[9].set(
            k_coeffs_torsional[3] * theta_x / cls.l
        )
        assert jnp.allclose(g_int_torsional, expected_f_torsional), (
            f"Torsional force calculation incorrect, expected {expected_f_torsional}, got {g_int_torsional}"
        )

    @classmethod
    def test_bending_strain_y(cls):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """

        eiy = 2.5

        k_coeffs_bending = jnp.full(6, 1e5).at[4].set(eiy)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )

        kappa_y = 1.0

        d_bending = jnp.zeros((1, 6))
        d_bending = d_bending.at[0, 0].set(cls.l)
        d_bending = d_bending.at[0, 4].set(kappa_y * cls.l)

        eps_bending = cls.struct.make_eps(d_bending)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, kappa_y, 0.0))
        assert jnp.allclose(eps_bending, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending}"
        )

        g_int = cls.struct.make_g_int_and_k_t(d_bending, True, True)[0]
        g_int_expected = jnp.zeros(12)
        g_int_expected = g_int_expected.at[4].set(-eiy * kappa_y)
        g_int_expected = g_int_expected.at[10].set(eiy * kappa_y)

        assert jnp.allclose(g_int, g_int_expected), (
            f"Bending force calculation incorrect, expected {g_int_expected}, got {g_int}"
        )

    @classmethod
    def test_bending_strain_z(cls):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """

        eiz = 2.5

        k_coeffs_bending = jnp.full(6, 1e5).at[5].set(eiz)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_bending)[None, :], None
        )

        kappa_z = 1.0

        d_bending = jnp.zeros((1, 6))
        d_bending = d_bending.at[0, 0].set(cls.l)
        d_bending = d_bending.at[0, 5].set(kappa_z * cls.l)

        eps_bending = cls.struct.make_eps(d_bending)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, kappa_z))
        assert jnp.allclose(eps_bending, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending}"
        )

        g_int = cls.struct.make_g_int_and_k_t(d_bending, True, True)[0]
        g_int_expected = jnp.zeros(12)
        g_int_expected = g_int_expected.at[5].set(-eiz * kappa_z)
        g_int_expected = g_int_expected.at[11].set(eiz * kappa_z)

        assert jnp.allclose(g_int, g_int_expected), (
            f"Bending force calculation incorrect, expected {g_int_expected}, got {g_int}"
        )

    @classmethod
    def test_solve_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces when solved for no external loads.
        """

        k_coeffs_unloaded = jnp.full(6, 4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_unloaded)[None, :], None
        )
        f_ext = jnp.zeros((1, 2, 6))
        ha, d, f_int = cls.struct.static_solve(f_ext, jnp.arange(6))

        assert jnp.allclose(ha, cls.struct.ha0), (
            f"Unloaded static solve contained deformation, expected {cls.struct.ha0}, got {ha}"
        )
        assert jnp.allclose(d, exp := jnp.array((cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))), (
            f"Incorrect configuration for unloaded static solve, expected {exp}, got {d}"
        )
        assert jnp.allclose(f_int, 0.0), (
            f"Internal force vector incorrect for unloaded static solve, expected zero force, got {f_int}"
        )

    @classmethod
    def test_solve_axial_load(cls):
        r"""
        Ensure axial load case is solved correctly.
        """
        k_coeffs_axial = jnp.full(6, 1e5).at[0].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_axial)[None, :], None
        )
        axial_load = 0.123
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 0].set(axial_load)
        ha, d, f_int = cls.struct.static_solve(f_ext, jnp.arange(6))

        expected_strain = axial_load / k_coeffs_axial[0]
        expected_disp = expected_strain * cls.l
        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l + expected_disp, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {d}"
        )
        assert jnp.allclose(f_int[:, 1:], 0.0), (
            f"Internal force vector expected to have zero shear/moment/torsion components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 0], -axial_load), (
            f"Internal axial force at fixed end incorrect, expected {-axial_load}, got {f_int[0, 0]}"
        )

        assert jnp.isclose(f_int[1, 0], axial_load), (
            f"Internal axial force at loaded end incorrect, expected {axial_load}, got {f_int[1, 0]}"
        )

    @classmethod
    def test_solve_torsional_load(cls):
        r"""
        Ensure torsion load case is solved correctly.
        """
        k_coeffs_torsion = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_torsion)[None, :], None
        )
        torsion_load = 0.123
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 3].set(torsion_load)
        ha, d, f_int = cls.struct.static_solve(f_ext, jnp.arange(6))

        expected_strain = torsion_load / k_coeffs_torsion[3]
        expected_disp = expected_strain * cls.l
        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l, 0.0, 0.0, expected_disp, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for torsional load static solve, expected {exp}, got {d}"
        )
        assert jnp.allclose(f_int[:, jnp.array((0, 1, 2, 4, 5))], 0.0), (
            f"Internal force vector expected to have zero shear/moment/axial components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 3], -torsion_load), (
            f"Internal axial force at fixed end incorrect, expected {-torsion_load}, got {f_int[0, 3]}"
        )

        assert jnp.isclose(f_int[1, 3], torsion_load), (
            f"Internal axial force at loaded end incorrect, expected {torsion_load}, got {f_int[1, 3]}"
        )

    @classmethod
    def test_solve_bending_y_load(cls):
        r"""
        Ensure z-bending load case is solved correctly.
        """
        k_coeffs_axial = jnp.full(6, 1e-3).at[4].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_axial)[None, :], None
        )
        bending_y_load = 1e-1
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 4].set(bending_y_load)
        ha, d, f_int = cls.struct.static_solve(f_ext, jnp.arange(6))

        expected_curvature = bending_y_load / k_coeffs_axial[4]
        expected_angle = expected_curvature * cls.l

        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l, 0.0, 0.0, 0.0, expected_angle, 0.0)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {d}"
        )

        assert jnp.allclose(f_int[:, jnp.array((0, 1, 2, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/moment/z-moment components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 4], -bending_y_load), (
            f"Internal bending moment at fixed end incorrect, expected {-bending_y_load}, got {f_int[0, 4]}"
        )

        assert jnp.isclose(f_int[1, 4], bending_y_load), (
            f"Internal bending moment at loaded end incorrect, expected {bending_y_load}, got {f_int[1, 4]}"
        )

        hg = vmap(exp_se3)(ha)
        coord_tip = hg[1, :3, 3]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, cls.l, direction="y"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending y load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    @classmethod
    def test_solve_bending_z_load(cls):
        r"""
        Ensure bending in z load case is solved correctly.
        """
        k_coeffs_axial = jnp.full(6, 1e-3).at[5].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_axial)[None, :], None
        )
        bending_z_load = 1e-1
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 5].set(bending_z_load)
        ha, d, f_int = cls.struct.static_solve(f_ext, jnp.arange(6))

        expected_curvature = bending_z_load / k_coeffs_axial[5]
        expected_angle = expected_curvature * cls.l

        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l, 0.0, 0.0, 0.0, 0.0, expected_angle)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {d}"
        )

        assert jnp.allclose(f_int[:, jnp.array((0, 1, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/y-moment components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 5], -bending_z_load), (
            f"Internal z-moment at fixed end incorrect, expected {-bending_z_load}, got {f_int[0, 5]}"
        )

        assert jnp.isclose(f_int[1, 5], bending_z_load), (
            f"Internal z-moment at loaded end incorrect, expected {bending_z_load}, got {f_int[1, 5]}"
        )

        hg = vmap(exp_se3)(ha)
        coord_tip = hg[1, :3, 3]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, cls.l, direction="z"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending z load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    @classmethod
    def test_solve_shear_y_load(cls):
        r"""
        Ensure shear load case is solved correctly.
        """
        k_coeffs_shear_y = jnp.full(6, 1e5).at[1].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_shear_y)[None, :], None
        )
        shear_y_load = 7.89
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 1].set(shear_y_load)
        ha, d, f_int = cls.struct.static_solve(
            f_ext, jnp.concatenate((jnp.arange(6), jnp.arange(9, 12)))
        )

        expected_strain = shear_y_load / k_coeffs_shear_y[1]
        expected_disp = expected_strain * cls.l
        expected_moment = -0.5 * shear_y_load * cls.l

        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l, expected_disp, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {d}"
        )
        assert jnp.allclose(f_int[:, jnp.array((0, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/z_shear components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 1], -shear_y_load), (
            f"Internal shear_y force at fixed end incorrect, expected {-shear_y_load}, got {f_int[0, 1]}"
        )

        assert jnp.isclose(f_int[1, 1], shear_y_load), (
            f"Internal shear_y force at loaded end incorrect, expected {shear_y_load}, got {f_int[1, 1]}"
        )

        assert jnp.isclose(f_int[0, 5], expected_moment), (
            f"Internal moment at fixed end incorrect, expected {expected_moment}, got {f_int[0, 5]}",
        )

        assert jnp.isclose(f_int[1, 5], expected_moment), (
            f"Internal moment at loaded end incorrect, expected {expected_moment}, got {f_int[1, 5]}"
        )

    @classmethod
    def test_solve_shear_z_load(cls):
        r"""
        Ensure shear load case is solved correctly.
        """
        k_coeffs_shear_z = jnp.full(6, 1e5).at[2].set(4.56)
        cls.struct.set_design_variables(
            cls.coords, jnp.diag(k_coeffs_shear_z)[None, :], None
        )
        shear_z_load = 7.89
        f_ext = jnp.zeros((1, 2, 6)).at[0, 1, 2].set(shear_z_load)
        ha, d, f_int = cls.struct.static_solve(
            f_ext, jnp.concatenate((jnp.arange(6), jnp.arange(9, 12)))
        )

        expected_strain = shear_z_load / k_coeffs_shear_z[2]
        expected_disp = expected_strain * cls.l
        expected_moment = 0.5 * shear_z_load * cls.l

        assert jnp.allclose(
            d,
            exp := jnp.array((cls.l, 0.0, expected_disp, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {d}"
        )
        assert jnp.allclose(f_int[:, jnp.array((0, 1, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/y-shear components, got {f_int}"
        )

        assert jnp.isclose(f_int[0, 2], -shear_z_load), (
            f"Internal shear_z force at fixed end incorrect, expected {-shear_z_load}, got {f_int[0, 2]}"
        )

        assert jnp.isclose(f_int[1, 2], shear_z_load), (
            f"Internal shear_z force at loaded end incorrect, expected {shear_z_load}, got {f_int[1, 2]}"
        )

        assert jnp.isclose(f_int[0, 4], expected_moment), (
            f"Internal moment at fixed end incorrect, expected {expected_moment}, got {f_int[0, 4]}",
        )

        assert jnp.isclose(f_int[1, 4], expected_moment), (
            f"Internal moment at loaded end incorrect, expected {expected_moment}, got {f_int[1, 4]}"
        )
