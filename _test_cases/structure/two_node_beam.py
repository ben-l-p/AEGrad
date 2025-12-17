from aegrad.structure.structure import Structure
from aegrad.algebra.test_routines import k_t_expected
from jax import numpy as jnp
from jax import Array
import numpy as np

coords = jnp.zeros((2, 3))
coords = coords.at[1, 0].set(1.0)
k_coeffs = jnp.arange(1, 7)
k_cs = jnp.diag(k_coeffs)

struct = Structure(2, jnp.array([[0, 1]]), jnp.array([[0.0, 1.0, 0.0]]))
struct.set_design_variables(coords, k_cs, None)

k = np.array(struct.make_k(struct.d0, include_material=True, include_geometric=False))
k_t_expected = np.array(k_t_expected(k_coeffs, struct.l0[0]))

pass
