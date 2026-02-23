from jax import numpy as jnp
import jax

from aegrad.algebra.so3 import log_so3
from models.geradin_beam import geradin_beam

jax.config.update("jax_enable_x64", True)


class TestGeradinBeam:
    struct = geradin_beam(20, "x")

    @classmethod
    def run_load_case(cls, load: float) -> tuple[float, float]:
        f_ext = jnp.zeros((cls.struct.n_nodes, 6))
        f_ext = f_ext.at[-1, 2].set(-load)

        result = cls.struct.static_solve(
            f_ext_follower=None,
            f_ext_dead=f_ext,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
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
