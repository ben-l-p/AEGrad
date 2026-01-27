from aegrad.structure.structure import Structure
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import numpy as np

v_init = jnp.array((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
l = 3.14
coords = jnp.zeros((2, 3)).at[1, 0].set(l)
conn = jnp.array([[0, 1]])
y_vect = jnp.array([[0.0, 1.0, 0.0]])
k_cs = jnp.diag(jnp.full(6, 1e9))
m_bar = 5.0 * jnp.eye(3)
j_bar = 0.1 * jnp.eye(3)
m_cs = block_diag(m_bar, j_bar)

n_tstep = 10
dt = 0.001

struct = Structure(2, conn, y_vect, None)
struct.set_design_variables(coords, k_cs, m_cs)

init_cond = struct.reference_configuration().to_dynamic()
init_cond.v = jnp.broadcast_to(v_init[None, :], (2, 6))

output = struct.dynamic_solve(init_cond, n_tstep, dt, None, None, None)
x_t = np.array(output.hg[:, 0, :3, 3])


pass
