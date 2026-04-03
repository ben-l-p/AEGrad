from copy import deepcopy
from jax import numpy as jnp
import jax
from jax import Array

from algebra.so3 import log_so3
from structure.beam import StaticStructure
from models.geradin_beam import geradin_beam

jax.config.update("jax_enable_x64", True)


class TestGeradinBeamADGradients:
    n_nodes = 20
    struct = geradin_beam(n_nodes, "x_target")
    load = 600000.0
    f_ext = jnp.zeros((n_nodes, 6)).at[-1, 2].set(-load)

    @staticmethod
    def get_tip_disp_rot(result: StaticStructure) -> Array:
        z_tip = result.hg[-1, 2, 3]
        rot = -log_so3(result.hg[-1, :3, :3])[1]

        return jnp.array((z_tip, rot))

    @classmethod
    def test_stiffness_gradient(cls):
        r"""
        Evaluate the gradients of the tip deflection and rotation with respect to the bending stiffness about the y-axis
        (eay) using both finite differences and automatic differentiation.
        """
        base_k_coeffs = jnp.diagonal(cls.struct.k_cs[0, ...])  # [6]
        base_eay = base_k_coeffs[4]  # bending stiffness about y-axis

        def func(eay: Array) -> Array:
            k_coeffs = base_k_coeffs.at[4].set(eay)
            this_struct = deepcopy(cls.struct)
            this_struct.set_design_variables(
                this_struct.x0, jnp.diag(k_coeffs), None, None
            )

            result = this_struct.static_solve(
                f_ext_follower=None,
                f_ext_dead=cls.f_ext,
                f_ext_aero=None,
                prescribed_dofs=jnp.arange(6),
                load_steps=3,
            )

            return cls.get_tip_disp_rot(result)

        eps = 10.0  # large epsilon as stiffnesses are large
        base_result = func(base_eay)
        perturb_result = func(base_eay + eps)
        grads_fd = (perturb_result - base_result) / eps

        grads_ad = jax.jacfwd(func)(base_eay)

        err = jnp.abs(grads_fd - grads_ad) / grads_ad

        assert err.max() < 1e-5, "Max relative error for stiffness gradient exceeded"

    @classmethod
    def test_coordinate_gradient(cls):
        r"""
        Evaluate the gradients of the tip deflection and rotation with respect to the b_ref of the beam using both
        finite differences and automatic differentiation.
        """
        base_coords = cls.struct.x0

        def func(length_fact: Array) -> Array:
            coords = base_coords * length_fact
            this_struct = deepcopy(cls.struct)
            this_struct.set_design_variables(coords, this_struct.k_cs, None, None)

            result = this_struct.static_solve(
                f_ext_follower=None,
                f_ext_dead=cls.f_ext,
                f_ext_aero=None,
                prescribed_dofs=jnp.arange(6),
                load_steps=3,
            )

            return cls.get_tip_disp_rot(result)

        eps = 1e-5
        base_result = func(jnp.array(1.0))
        perturb_result = func(jnp.array(1.0 + eps))
        grads_fd = (perturb_result - base_result) / eps

        grads_ad = jax.jacfwd(func)(jnp.array(1.0))

        err = jnp.abs(grads_fd - grads_ad) / grads_ad

        assert err.max() < 1e-5, "Max relative error for coordinate gradient exceeded"

    @classmethod
    def test_force_gradient(cls):
        r"""
        Evaluate the gradients of the tip deflection and rotation with respect to the applied tip load using both
        finite differences and automatic differentiation.
        """

        def func(f_fact: Array) -> Array:
            this_struct = deepcopy(cls.struct)
            result = this_struct.static_solve(
                f_ext_follower=None,
                f_ext_dead=cls.f_ext * f_fact,
                f_ext_aero=None,
                prescribed_dofs=jnp.arange(6),
                load_steps=3,
            )

            return cls.get_tip_disp_rot(result)

        eps = 1e-5
        base_result = func(jnp.array(1.0))
        perturb_result = func(jnp.array(1.0 + eps))
        grads_fd = (perturb_result - base_result) / eps

        grads_ad = jax.jacfwd(func)(jnp.array(1.0))

        err = jnp.abs(grads_fd - grads_ad) / grads_ad

        assert err.max() < 1e-5, "Max relative error for force gradient exceeded"
