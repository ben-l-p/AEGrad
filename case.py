from jax import numpy as jnp

from models.cantilever_wing import make_cantilever_wing

wing = make_cantilever_wing(
    n_nodes=50,
    m=20,
    m_star=10,
    u_inf=jnp.array((6.0, 0.0, 1.0)),
)

result = wing.static_solve(
    f_ext_dead=None,
    f_ext_follower=None,
    horseshoe=True,
    prescribed_dofs=jnp.arange(6),
)

result.plot("./")

# d_zeta_b = np.array(
#     _d_gamma_b_d_zeta_b(
#         gamma_bs=result.aero.gamma_b,
#         gamma_ws=result.aero.gamma_w,
#         zeta_bs=result.aero.zeta_b,
#         zeta_ws=result.aero.zeta_w,
#         kernels=[*wing.aero.kernels_b, *wing.aero.kernels_w],
#         flowfield=wing.aero.flowfield,
#         mirror_normal=wing.aero.mirror_normal,
#         mirror_point=wing.aero.mirror_point,
#         t=result.aero.t,
#         static=True,
#     )
# )

pass
