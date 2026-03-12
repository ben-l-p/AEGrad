from jax import numpy as jnp
import numpy as np

from models.cantilever_wing import make_cantilever_wing

wing = make_cantilever_wing(
    n_nodes=20,
    m=6,
    m_star=15,
    u_inf=jnp.array((6.0, 0.0, 1.0)),
)

result = wing.static_solve(
    f_ext_dead=None,
    f_ext_follower=None,
    horseshoe=True,
    prescribed_dofs=jnp.arange(6),
)

d_gamma_d_zeta = np.array(result.aero.d_gamma_b_d_zeta_b(static=True, i_ts=0))

pass
