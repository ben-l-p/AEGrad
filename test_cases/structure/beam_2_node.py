from aegrad.structure.structure import Structure
from jax import numpy as jnp
import numpy as np
from aegrad.algebra.test_routines import k_t_expected

n_nodes: int = 2
n_elem: int = n_nodes - 1
l_beam: float = 1.0

conn = jnp.zeros((n_elem, 2), dtype=int)    # [n_elem, 2]
conn = conn.at[:, 0].set(jnp.arange(0, n_elem))
conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

normal_vector = jnp.zeros((n_elem, 3), dtype=float)
normal_vector = normal_vector.at[:, 2].set(1.0)

# create data structure
struct = Structure(n_nodes, conn, normal_vector)

x0 = jnp.zeros((n_nodes, 3)) # [n_node, 3]
x0 = x0.at[:, 0].set(jnp.linspace(0, l_beam, n_nodes))

# stiffness
eax, gay, gaz = 1.0, 0.0, 0.0
gjx, eiy, eiz = 1.0, 0.0, 0.0
k_diag = jnp.array((eax, gay, gaz, gjx, eiy, eiz))
k_cs = jnp.diag(k_diag)    # [6, 6]

# add design variables
struct.set_design_variables(x0, k_cs, None)

# same for EAx and GJx
# shear is not included in this routine
# differences for EIy an EIz
k_t = np.array(struct.make_k_t(struct.d0, False))

k_t_expected = np.array(k_t_expected(k_diag, l_beam))

k_t_diff = k_t - k_t_expected
if_diff = not jnp.allclose(k_t_diff, 0.0)

pass