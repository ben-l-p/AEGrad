from jax import numpy as jnp
import jax
from jax import Array

from aegrad.aero.uvlm import UVLM
from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.flowfields import Constant

jax.config.update("jax_enable_x64", True)


def test_vlm_gradients():
    gd = GridDiscretization(m=4, n=6, m_star=5)
    flow = Constant(u_inf=jnp.array((1.0, 0.0, 0.1)), rho=1.225, relative_motion=True)

    c_ref = 1.0
    b_ref = 5.0

    zeta = jnp.zeros((gd.m + 1, gd.n + 1, 3))
    zeta = zeta.at[:, :, 0].set(jnp.linspace(0.0, c_ref, gd.m + 1)[:, None])
    zeta = zeta.at[:, :, 1].set(jnp.linspace(0.0, b_ref, gd.n + 1)[None, :])

    # Use identity hg so that zeta_b = x0_aero = zeta directly.
    hg_identity = jnp.zeros((gd.n + 1, 4, 4)).at[...].set(jnp.eye(4)[None, ...])

    case = UVLM(grid_shapes=[gd], dof_mapping=jnp.arange(gd.n + 1))
    case.set_design_variables(
        dt=0.1, flowfield=flow, x0_aero=zeta, hg0=hg_identity, delta_w=None
    )
    aero_sol = case.solve_static()

    case_ = UVLM(grid_shapes=[gd], dof_mapping=jnp.arange(gd.n + 1))

    def gamma_from_zeta(zeta_: Array) -> Array:
        hg = jnp.zeros((gd.n + 1, 4, 4)).at[...].set(jnp.eye(4)[None, ...])
        case_.set_design_variables(
            dt=0.1, flowfield=flow, x0_aero=zeta_, hg0=hg, delta_w=None
        )
        aero_sol_ = case_.solve_static()
        return aero_sol_.gamma_b[0]

    d_gamma_d_zeta, d_f_d_zeta = aero_sol.static_d_sol_d_zeta_b(
        i_ts=0, include_midpoint_velocity=True
    )

    d_gamma_d_zeta_ad = jax.jacrev(gamma_from_zeta)(zeta).reshape(d_gamma_d_zeta.shape)

    assert jnp.allclose(d_gamma_d_zeta_ad, d_gamma_d_zeta), (
        "Circulation Jacobian does not match AD"
    )

    def f_steady_from_zeta(zeta_: Array) -> Array:
        hg = jnp.zeros((gd.n + 1, 4, 4)).at[...].set(jnp.eye(4)[None, ...])
        case_.set_design_variables(
            dt=0.1, flowfield=flow, x0_aero=zeta_, hg0=hg, delta_w=None
        )
        aero_sol_ = case_.solve_static()
        return aero_sol_.f_steady[0].ravel()

    d_f_d_zeta_ad = jax.jacrev(f_steady_from_zeta)(zeta).reshape(d_f_d_zeta.shape)

    assert jnp.allclose(d_f_d_zeta_ad, d_f_d_zeta), "Forcing Jacobian does not match AD"
