from jax import numpy as jnp
import jax

from aegrad.structure import BeamStructure
from aegrad.algebra.base import chi
from aegrad.algebra.test_routines import const_curvature_beam
from aegrad.algebra.se3 import p

jax.config.update("jax_enable_x64", True)


class TestTwoNodeXBeamStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    beam_direction = "x"
    direction_index = 0
    y_vector = jnp.array([[0.0, 1.0, 0.0]])

    length = jnp.array(3.45)
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(length)
    struct = BeamStructure(2, jnp.array([[0, 1]]), y_vector)

    @classmethod
    def test_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """

        k_coeffs = jnp.ones(6)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        d = jnp.zeros((1, 6)).at[0, 0].set(cls.length)
        eps = cls.struct._make_eps(d)
        assert jnp.allclose(eps, 0.0), (
            f"Axial strain calculation incorrect, expected zero strain, got {eps}"
        )
        f_int = cls.struct._make_f_int(
            p(d[0, :], cls.struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        assert jnp.allclose(f_int, 0.0), (
            f"Internal force vector incorrect, expected zero force, got {f_int}"
        )

    @classmethod
    def test_axial_strain(cls):
        r"""
        Ensure axial strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(cls.length + dx)
        eps = cls.struct._make_eps(d)
        expected_eps = jnp.array((dx / cls.length, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Axial strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = cls.struct._make_f_int(
            p(d[0, :], cls.struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[0].set(k_coeffs[0] * dx / cls.length)
        expected_f_int = expected_f_int.at[6].set(-k_coeffs[0] * dx / cls.length)
        expected_f_int = chi(chi(cls.struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Axial force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    @classmethod
    def test_torsional_strain(cls):
        r"""
        Ensure torsional strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        theta_x = 0.1
        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(cls.length)
        d = d.at[0, 3].set(theta_x)
        eps = cls.struct._make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, theta_x / cls.length, 0.0, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Torsional strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = cls.struct._make_f_int(
            p(d[0, :], cls.struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[3].set(k_coeffs[3] * theta_x / cls.length)
        expected_f_int = expected_f_int.at[9].set(-k_coeffs[3] * theta_x / cls.length)
        expected_f_int = chi(chi(cls.struct.o0[0, ...])) @ expected_f_int
        assert jnp.allclose(f_int, expected_f_int), (
            f"Torsional force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    @classmethod
    def test_bending_strain_y(cls):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """

        eiy = 2.5

        k_coeffs = jnp.full(6, 1e5).at[4].set(eiy)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)

        kappa_y = 1.0

        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(cls.length)
        d = d.at[0, 4].set(kappa_y * cls.length)

        eps = cls.struct._make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, kappa_y, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )

        f_int = cls.struct._make_f_int(
            p(d[0, :], cls.struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[4].set(eiy * kappa_y)
        expected_f_int = expected_f_int.at[10].set(-eiy * kappa_y)
        expected_f_int = chi(chi(cls.struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Bending force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    @classmethod
    def test_bending_strain_z(cls):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """

        eiz = 2.5

        k_coeffs = jnp.full(6, 1e5).at[5].set(eiz)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)

        kappa_z = 1.0

        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(cls.length)
        d = d.at[0, 5].set(kappa_z * cls.length)

        eps = cls.struct._make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, kappa_z))
        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )

        f_int = cls.struct._make_f_int(
            p(d[0, :], cls.struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[5].set(eiz * kappa_z)
        expected_f_int = expected_f_int.at[11].set(-eiz * kappa_z)
        expected_f_int = chi(chi(cls.struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Bending force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    @classmethod
    def test_solve_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces when solved for no external loads.
        """

        k_coeffs = jnp.full(6, 4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        result = cls.struct.static_solve(
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        assert jnp.allclose(result.hg, cls.struct.hg0), (
            f"Unloaded static solve contained deformation, expected {cls.struct.hg0}, got {result.hg}"
        )
        assert jnp.allclose(
            result.d, exp := jnp.array((cls.length, 0.0, 0.0, 0.0, 0.0, 0.0))
        ), (
            f"Incorrect configuration for unloaded static solve, expected {exp}, got {result.d}"
        )
        assert jnp.allclose(result.f_int, 0.0), (
            f"Internal force vector incorrect for unloaded static solve, expected zero force, got {result.f_int}"
        )

    @classmethod
    def test_solve_axial_load(cls):
        r"""
        Ensure axial load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 0.123
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(cls.struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_eps = load / k_coeffs[0]
        expected_disp = expected_eps * cls.length
        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length + expected_disp, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, 1:], 0.0), (
            f"Internal force vector expected to have zero shear/moment/torsion components, got {f_int_rot[:, 1:]}"
        )

        assert jnp.isclose(f_int_rot[0, 0], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 0]}"
        )

        assert jnp.isclose(f_int_rot[1, 0], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 0]}"
        )

    @classmethod
    def test_solve_torsional_load(cls):
        r"""
        Ensure torsion load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 0.123
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(cls.struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_strain = load / k_coeffs[3]
        expected_disp = expected_strain * cls.length
        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length, 0.0, 0.0, expected_disp, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for torsional load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 4, 5))], 0.0), (
            f"Internal force vector expected to have zero shear/moment/axial components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 3], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 3]}"
        )

        assert jnp.isclose(f_int_rot[1, 3], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 3]}"
        )

    @classmethod
    def test_solve_bending_y_load(cls):
        r"""
        Ensure z-bending load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e-3).at[4].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1e-1
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(cls.struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_curvature = load / k_coeffs[4]
        expected_angle = expected_curvature * cls.length

        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length, 0.0, 0.0, 0.0, expected_angle, 0.0)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/moment/z-moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 4], load), (
            f"Internal bending moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 4]}"
        )

        assert jnp.isclose(f_int_rot[1, 4], -load), (
            f"Internal bending moment at loaded end incorrect, expected {-load}, got {f_int_rot[1, 4]}"
        )

        coord_tip = cls.struct.o0[0, ...].T @ result.hg[1, :3, 3]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, cls.length, direction="y"
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
        k_coeffs = jnp.full(6, 1e-3).at[5].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1e-1
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(cls.struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_curvature = load / k_coeffs[5]
        expected_angle = expected_curvature * cls.length

        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length, 0.0, 0.0, 0.0, 0.0, expected_angle)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/y-moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 5], load), (
            f"Internal z-moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 5]}"
        )

        assert jnp.isclose(f_int_rot[1, 5], -load), (
            f"Internal z-moment at loaded end incorrect, expected {-load}, got {f_int_rot[1, 5]}"
        )

        coord_tip = cls.struct.o0[0, ...].T @ result.hg[1, :3, 3]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, cls.length, direction="z"
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
        k_coeffs = jnp.full(6, 1e5).at[1].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 7.89
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(cls.struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate((jnp.arange(6), jnp.arange(9, 12))),
        )

        expected_strain = load / k_coeffs[1]
        expected_disp = expected_strain * cls.length
        expected_moment = -0.5 * load * cls.length

        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length, expected_disp, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/z_shear components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 1], load), (
            f"Internal shear_y force at fixed end incorrect, expected {load}, got {f_int_rot[0, 1]}"
        )

        assert jnp.isclose(f_int_rot[1, 1], -load), (
            f"Internal shear_y force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 1]}"
        )

        assert jnp.isclose(f_int_rot[0, 5], -expected_moment), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 5]}",
        )

        assert jnp.isclose(f_int_rot[1, 5], -expected_moment), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {result.f_int[1, 5]}"
        )

    @classmethod
    def test_solve_shear_z_load(cls):
        r"""
        Ensure shear load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[2].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 7.89
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(cls.struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = cls.struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate((jnp.arange(6), jnp.arange(9, 12))),
        )

        expected_strain = load / k_coeffs[2]
        expected_disp = expected_strain * cls.length
        expected_moment = 0.5 * load * cls.length

        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.length, 0.0, expected_disp, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(cls.struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/y-shear components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 2], load), (
            f"Internal shear_z force at fixed end incorrect, expected {load}, got {f_int_rot[0, 2]}"
        )

        assert jnp.isclose(f_int_rot[1, 2], -load), (
            f"Internal shear_z force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 2]}"
        )

        assert jnp.isclose(f_int_rot[0, 4], -expected_moment), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 4]}",
        )

        assert jnp.isclose(f_int_rot[1, 4], -expected_moment), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {f_int_rot[1, 4]}"
        )


class TestTwoNodeYBeamStrainsForces(TestTwoNodeXBeamStrainsForces):
    beam_direction = "y"
    direction_index = 1
    y_vector = jnp.array([[0.0, 0.0, 1.0]])

    length = jnp.array(3.45)
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(length)
    struct = BeamStructure(2, jnp.array([[0, 1]]), y_vector)


class TestTwoNodeZBeamStrainsForces(TestTwoNodeXBeamStrainsForces):
    beam_direction = "z"
    direction_index = 2
    y_vector = jnp.array([[1.0, 0.0, 0.0]])

    length = jnp.array(3.45)
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(length)
    struct = BeamStructure(2, jnp.array([[0, 1]]), y_vector)
