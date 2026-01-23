from jax import numpy as jnp
import jax
from aegrad.structure.structure import Structure
from aegrad.algebra.test_routines import const_curvature_beam
from aegrad.algebra.base import chi

jax.config.update("jax_enable_x64", True)


class TestMultiXElementStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    l = jnp.array(2.0)
    n_nodes = 10
    n_elem = n_nodes - 1
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    beam_direction = "x"
    direction_index = 0
    coords = (
        jnp.zeros((n_nodes, 3)).at[:, direction_index].set(jnp.linspace(0, l, n_nodes))
    )
    y_vect = jnp.zeros((n_elem, 3)).at[:, 1].set(1.0)
    struct = Structure(n_nodes, conn, y_vect)

    @classmethod
    def test_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """

        k_coeffs = jnp.full(6, 4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        d = jnp.zeros((cls.n_elem, 6)).at[:, 0].set(cls.l / cls.n_elem)
        strain = cls.struct.make_eps(d)
        assert jnp.allclose(strain, 0.0), (
            f"Strain calculation incorrect, expected zero strain, got {strain}"
        )
        f_int = cls.struct.make_f_int_and_k_t(d, True, True)[0]
        assert jnp.allclose(f_int, 0.0), (
            f"Internal force vector incorrect, expected zero, got {f_int}"
        )

    @classmethod
    def test_axial_strain(cls):
        r"""
        Ensure axial strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((cls.n_elem, 6))
        d = d.at[:, 0].set((cls.l + dx) / cls.n_elem)
        eps = cls.struct.make_eps(d)
        expected_eps = jnp.array((dx / cls.l, 0.0, 0.0, 0.0, 0.0, 0.0))[None, :]
        expected_f = k_coeffs[0] * dx / cls.l

        assert jnp.allclose(eps, expected_eps), (
            f"Axial strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = cls.struct.make_f_int_and_k_t(d, True, True)[0].reshape(cls.n_nodes, 6)

        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 0], -expected_f), (
            f"Axial force calculation at root incorrect, expected {-expected_f}, got {f_int_rot[0, 0]}"
        )

        assert jnp.allclose(f_int_rot[-1, 0], expected_f), (
            f"Axial force calculation at tip incorrect, expected {expected_f}, got {f_int_rot[-1, 0]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {0, 6 * (cls.n_nodes - 1)})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Axial force in beam incorrect, expected zero, got {f_int_rot}"
        )

    @classmethod
    def test_torsional_strain(cls):
        r"""
        Ensure torsional strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((cls.n_elem, 6))
        d = d.at[:, 0].set(cls.l / cls.n_elem)
        d = d.at[:, 3].set(dx / cls.n_elem)

        strain = cls.struct.make_eps(d)
        expected_strain = jnp.array((0.0, 0.0, 0.0, dx / cls.l, 0.0, 0.0))[None, :]
        expected_f = k_coeffs[3] * dx / cls.l

        assert jnp.allclose(strain, expected_strain), (
            f"Torsional strain calculation incorrect, expected {expected_strain}, got {strain}"
        )
        f_int = cls.struct.make_f_int_and_k_t(d, True, True)[0].reshape(cls.n_nodes, 6)

        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 3], -expected_f), (
            f"Torsional force calculation at root incorrect, expected {-expected_f}, got {f_int_rot[0, 3]}"
        )

        assert jnp.allclose(f_int_rot[-1, 3], expected_f), (
            f"Torsional force calculation at tip incorrect, expected {expected_f}, got {f_int_rot[-1, 3]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {3, 6 * (cls.n_nodes - 1) + 3})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    @classmethod
    def test_bending_y_strain(cls):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[4].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((cls.n_elem, 6))
        d = d.at[:, 0].set(cls.l / cls.n_elem)
        d = d.at[:, 4].set(dx / cls.n_elem)

        eps_bending = cls.struct.make_eps(d)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, dx / cls.l, 0.0))[
            None, :
        ]
        expected_f = k_coeffs[4] * dx / cls.l

        assert jnp.allclose(eps_bending, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps_bending}"
        )
        f_int = cls.struct.make_f_int_and_k_t(d, True, True)[0].reshape(cls.n_nodes, 6)
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 4], -expected_f), (
            f"Bending moment calculation at root incorrect, expected {-expected_f}, got {f_int_rot[0, 4]}"
        )

        assert jnp.allclose(f_int_rot[-1, 4], expected_f), (
            f"Bending calculation at tip incorrect, expected {expected_f}, got {f_int_rot[-1, 4]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {4, 6 * (cls.n_nodes - 1) + 4})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    @classmethod
    def test_bending_z_strain(cls):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[5].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((cls.n_elem, 6))
        d = d.at[:, 0].set(cls.l / cls.n_elem)
        d = d.at[:, 5].set(dx / cls.n_elem)

        eps = cls.struct.make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, dx / cls.l))[None, :]
        expected_f = k_coeffs[5] * dx / cls.l

        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = cls.struct.make_f_int_and_k_t(d, True, True)[0].reshape(cls.n_nodes, 6)
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 5], -expected_f), (
            f"Bending moment calculation at root incorrect, expected {-expected_f}, got {f_int_rot[0, 5]}"
        )

        assert jnp.allclose(f_int_rot[-1, 5], expected_f), (
            f"Bending calculation at tip incorrect, expected {expected_f}, got {f_int_rot[-1, 5]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * cls.n_nodes)) - {5, 6 * (cls.n_nodes - 1) + 5})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    @classmethod
    def test_solve_unloaded(cls):
        r"""
        Ensure undeformed beam has zero strains and internal forces when solved for no external loads.
        """

        k_coeffs = jnp.full(6, 4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        f_ext = jnp.zeros((cls.n_elem, 2, 6))
        result = cls.struct.static_solve(f_ext, None, jnp.arange(6))

        assert jnp.allclose(result.hg, cls.struct.hg0), (
            f"Unloaded static solve contained deformation, expected {cls.struct.hg0}, got {result.hg}"
        )
        assert jnp.allclose(
            result.d,
            exp := jnp.array((cls.l / cls.n_elem, 0.0, 0.0, 0.0, 0.0, 0.0))[None, :],
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
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 0].set(load)
        result = cls.struct.static_solve(f_ext, None, jnp.arange(6))

        expected_eps = load / k_coeffs[0]
        expected_disp = expected_eps * cls.l
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                ((cls.l + expected_disp) / cls.n_elem, 0.0, 0.0, 0.0, 0.0, 0.0)
            )[None, :],
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(cls.n_nodes * 6)) - {0, (cls.n_nodes - 1) * 6})
        )

        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero shear/moment/torsion components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 0], -load), (
            f"Internal axial force at fixed end incorrect, expected {-load}, got {f_int_rot[0, 0]}"
        )

        assert jnp.isclose(f_int_rot[-1, 0], load), (
            f"Internal axial force at loaded end incorrect, expected {load}, got {f_int_rot[-1, 0]}"
        )

    @classmethod
    def test_solve_torsional_load(cls):
        r"""
        Ensure torsional load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 3].set(load)
        result = cls.struct.static_solve(f_ext, None, jnp.arange(6))

        expected_strain = load / k_coeffs[3]
        expected_disp = expected_strain * cls.l
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (cls.l / cls.n_elem, 0.0, 0.0, expected_disp / cls.n_elem, 0.0, 0.0)
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(cls.n_nodes * 6)) - {3, (cls.n_nodes - 1) * 6 + 3})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 3], -load), (
            f"Internal axial force at fixed end incorrect, expected {-load}, got {f_int_rot[0, 3]}"
        )

        assert jnp.isclose(f_int_rot[-1, 3], load), (
            f"Internal axial force at loaded end incorrect, expected {load}, got {f_int_rot[-1, 3]}"
        )

    @classmethod
    def test_solve_y_bending_load(cls):
        r"""
        Ensure y-bending load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[4].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 4].set(load)
        result = cls.struct.static_solve(f_ext, None, jnp.arange(6))

        expected_strain = load / k_coeffs[4]
        expected_disp = expected_strain * cls.l
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (cls.l / cls.n_elem, 0.0, 0.0, 0.0, expected_disp / cls.n_elem, 0.0)
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(cls.n_nodes * 6)) - {4, (cls.n_nodes - 1) * 6 + 4})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/torsional components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 4], -load), (
            f"Internal moment at fixed end incorrect, expected {-load}, got {f_int_rot[0, 4]}"
        )

        assert jnp.isclose(f_int_rot[-1, 4], load), (
            f"Internal moment at loaded end incorrect, expected {load}, got {f_int_rot[-1, 4]}"
        )

        coord_tip = cls.struct.o0[-1, :, :].T @ result.hg[-1, :3, 3]

        expected_coord_tip = const_curvature_beam(expected_strain, cls.l, direction="y")

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending y load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    @classmethod
    def test_solve_z_bending_load(cls):
        r"""
        Ensure z-bending load case is solved correctly.
        """
        k_coeffs = jnp.full(6, 1e5).at[5].set(4.56)
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 5].set(load)
        result = cls.struct.static_solve(f_ext, None, jnp.arange(6))

        expected_strain = load / k_coeffs[5]
        expected_disp = expected_strain * cls.l
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (cls.l / cls.n_elem, 0.0, 0.0, 0.0, 0.0, expected_disp / cls.n_elem)
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(cls.n_nodes * 6)) - {5, (cls.n_nodes - 1) * 6 + 5})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/torsional components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 5], -load), (
            f"Internal moment at fixed end incorrect, expected {-load}, got {f_int_rot[0, 5]}"
        )

        assert jnp.isclose(f_int_rot[-1, 5], load), (
            f"Internal moment at loaded end incorrect, expected {load}, got {f_int_rot[-1, 5]}"
        )

        coord_tip = cls.struct.o0[-1, :, :].T @ result.hg[-1, :3, 3]

        expected_coord_tip = const_curvature_beam(expected_strain, cls.l, direction="z")

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
        Ensure y-shear load case is solved correctly.
        Note - this should not be expected to converge for z-bending moment, and as such has weaker tolerances.
        """

        k_coeffs = jnp.full(6, 1e5).at[1].set(4.56)
        k_coeffs = k_coeffs.at[0].set(1e2)  # allow the beam to stretch
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 1].set(load)
        result = cls.struct.static_solve(
            f_ext,
            None,
            jnp.concatenate((jnp.arange(6), (cls.n_nodes - 1) * 6 + jnp.arange(3, 6))),
            include_material=True,
            include_geometric=False,
            load_steps=4,
        )

        expected_eps = load / k_coeffs[1]
        expected_disp = expected_eps * cls.l
        expected_moment = -0.5 * load * cls.l

        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (cls.l / cls.n_elem, expected_disp / cls.n_elem, 0.0, 0.0, 0.0, 0.0)
            )[None, :],
            atol=1e-5,
            rtol=1e-3,
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.isclose(f_int_rot[0, 1], -load, atol=1e-4), (
            f"Internal shear_y force at fixed end incorrect, expected {-load}, got {f_int_rot[0, 1]}"
        )

        assert jnp.isclose(f_int_rot[-1, 1], load, atol=1e-4), (
            f"Internal shear_y force at loaded end incorrect, expected {load}, got {f_int_rot[-1, 1]}"
        )

        assert jnp.isclose(f_int_rot[0, 5], expected_moment, atol=1e-3), (
            f"Internal moment at fixed end incorrect, expected {expected_moment}, got {f_int_rot[0, 5]}",
        )

        assert jnp.isclose(f_int_rot[-1, 5], expected_moment, atol=1e-3), (
            f"Internal moment at loaded end incorrect, expected {expected_moment}, got {f_int_rot[-1, 5]}"
        )

    @classmethod
    def test_solve_shear_z_load(cls):
        r"""
        Ensure z-shear load case is solved correctly.
        Note - this should not be expected to converge for z-bending moment, and as such has weaker tolerances.
        """

        k_coeffs = jnp.full(6, 1e5).at[2].set(4.56)
        k_coeffs = k_coeffs.at[0].set(1e2)  # allow the beam to stretch
        cls.struct.set_design_variables(cls.coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = jnp.zeros((cls.n_elem, 2, 6)).at[-1, 1, 2].set(load)
        result = cls.struct.static_solve(
            f_ext,
            None,
            jnp.concatenate((jnp.arange(6), (cls.n_nodes - 1) * 6 + jnp.arange(3, 6))),
            include_material=True,
            include_geometric=False,
            load_steps=4,
        )

        expected_eps = load / k_coeffs[2]
        expected_disp = expected_eps * cls.l
        expected_moment = 0.5 * load * cls.l

        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (cls.l / cls.n_elem, 0.0, expected_disp / cls.n_elem, 0.0, 0.0, 0.0)
            )[None, :],
            atol=1e-5,
            rtol=1e-3,
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(cls.struct.o0[0, ...].T), f_int)

        assert jnp.isclose(f_int_rot[0, 2], -load, atol=1e-4), (
            f"Internal shear_y force at fixed end incorrect, expected {-load}, got {f_int_rot[0, 2]}"
        )

        assert jnp.isclose(f_int_rot[-1, 2], load, atol=1e-4), (
            f"Internal shear_y force at loaded end incorrect, expected {load}, got {f_int_rot[-1, 2]}"
        )

        assert jnp.isclose(f_int_rot[0, 4], expected_moment, atol=1e-3), (
            f"Internal moment at fixed end incorrect, expected {expected_moment}, got {f_int_rot[0, 4]}",
        )

        assert jnp.isclose(f_int_rot[-1, 4], expected_moment, atol=1e-3), (
            f"Internal moment at loaded end incorrect, expected {expected_moment}, got {f_int_rot[-1, 4]}"
        )


class TestMultiYElementStrainsForces(TestMultiXElementStrainsForces):
    n_nodes = TestMultiXElementStrainsForces.n_nodes
    n_elem = TestMultiXElementStrainsForces.n_elem
    conn = TestMultiXElementStrainsForces.conn
    l = 2.5

    beam_direction = "y"
    direction_index = 1
    coords = (
        jnp.zeros((n_nodes, 3)).at[:, direction_index].set(jnp.linspace(0, l, n_nodes))
    )
    y_vect = jnp.zeros((n_elem, 3)).at[:, 2].set(1.0)
    struct = Structure(n_nodes, conn, y_vect)


class TestMultiZElementStrainsForces(TestMultiXElementStrainsForces):
    n_nodes = TestMultiXElementStrainsForces.n_nodes
    n_elem = TestMultiXElementStrainsForces.n_elem
    conn = TestMultiXElementStrainsForces.conn
    l = 2.5

    beam_direction = "z"
    direction_index = 2
    coords = (
        jnp.zeros((n_nodes, 3)).at[:, direction_index].set(jnp.linspace(0, l, n_nodes))
    )
    y_vect = jnp.zeros((n_elem, 3)).at[:, 0].set(1.0)
    struct = Structure(n_nodes, conn, y_vect)
