from jax import numpy as jnp
from structure.structure import Structure
from algebra.so3 import vec_to_skew
import jax

jax.config.update("jax_enable_x64", True)


class TestStiffness:
    r"""
    Check the tangent stiffness matrix for undeformed straight beam
    """

    @staticmethod
    def test_undeformed_straight_beam():
        k_uu = jnp.linspace(0.1, 0.9, 9).reshape(3, 3)
        k_ww = jnp.linspace(1.1, 1.9, 9).reshape(3, 3)
        k_cs = jnp.block([[k_uu, jnp.zeros((3, 3))], [jnp.zeros((3, 3)), k_ww]])
        l = 3.45

        x = jnp.zeros((2, 3))
        x = x.at[1, 0].set(l)

        conn = jnp.array(((0, 1),))
        y_vect = jnp.array(((0.0, 1.0, 0.0),))

        struct = Structure(2, conn, y_vect)
        struct.set_design_variables(x, k_cs[None, ...], None)

        du0 = jnp.zeros(6).at[0].set(l)

        g_int, k_t = struct.make_f_int_and_k_t(du0[None, :], True, True)

        du0t = vec_to_skew(du0[:3])
        k_t_exp = (
            jnp.block(
                [
                    [k_uu, -0.5 * k_uu @ du0t, -k_uu, -0.5 * k_uu @ du0t],
                    [
                        0.5 * du0t @ k_uu,
                        k_ww - 0.25 * du0t @ k_uu @ du0t,
                        -0.5 * du0t @ k_uu,
                        -k_ww - 0.25 * du0t @ k_uu @ du0t,
                    ],
                    [-k_uu, 0.5 * k_uu @ du0t, k_uu, 0.5 * k_uu @ du0t],
                    [
                        0.5 * du0t @ k_uu,
                        -k_ww - 0.25 * du0t @ k_uu @ du0t,
                        -0.5 * du0t @ k_uu,
                        k_ww - 0.25 * du0t @ k_uu @ du0t,
                    ],
                ]
            )
            / l
        )

        assert jnp.allclose(k_t, k_t_exp), (
            "Tangent stiffness matrix for undeformed straight beam does not match analytical result"
        )
