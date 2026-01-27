from jax import numpy as jnp
import jax
from aegrad.structure.structure import Structure
from aegrad.algebra.so3 import log_so3

jax.config.update("jax_enable_x64", True)


class TestGeradinBeam:
    l = jnp.array(5.0)
    n_nodes = 20
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

    k_coeffs = jnp.full(6, 1e15)
    k_coeffs = k_coeffs.at[1:3].set(3.231e8)
    k_coeffs = k_coeffs.at[4:6].set(9.345e6)
    struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)

    @classmethod
    def run_load_case(cls, load: float) -> tuple[float, float]:
        f_ext = jnp.zeros((cls.n_nodes, 6))
        f_ext = f_ext.at[-1, 2].set(-load)

        result = cls.struct.static_solve(
            None,
            f_ext,
            jnp.arange(6),
            include_material=True,
            include_geometric=False,
            load_steps=3,
        )

        z_tip = result.hg[-1, 2, 3]
        rot = -log_so3(result.hg[-1, :3, :3])[1]

        return float(z_tip), float(rot)

    @classmethod
    def test_small_load(cls):
        z_tip, rot = cls.run_load_case(600.0)
        expected_z_tip = -2.6819e-3
        expected_rot = -8.025680e-4
        assert jnp.isclose(z_tip, expected_z_tip, rtol=1e-2), (
            f"Expected tip deflection {expected_z_tip}, got {z_tip}"
        )
        assert jnp.isclose(rot, expected_rot, rtol=1e-2), (
            f"Expected tip rotation {expected_rot}, got {rot}"
        )

    @classmethod
    def test_large_load(cls):
        z_tip, rot = cls.run_load_case(600000.0)
        expected_z_tip = -2.157409
        expected_rot = -6.721341e-1
        assert jnp.isclose(z_tip, expected_z_tip, rtol=1e-2), (
            f"Expected tip deflection {expected_z_tip}, got {z_tip}"
        )
        assert jnp.isclose(rot, expected_rot, rtol=1e-2), (
            f"Expected tip rotation {expected_rot}, got {rot}"
        )
