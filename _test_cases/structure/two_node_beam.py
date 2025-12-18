import jax
from aegrad.structure.structure import Structure
from aegrad.algebra.test_routines import k_t_expected
from jax import numpy as jnp
from jax import Array
import numpy as np

jax.enable_x64()

l = 3.14
coords = jnp.zeros((2, 3))
coords = coords.at[1, 0].set(l)

struct = Structure(2, jnp.array([[0, 1]]), jnp.array([[0.0, 1.0, 0.0]]))

# check no load case - working
k_coeffs_unloaded = jnp.ones(6)
struct.set_design_variables(coords, jnp.diag(k_coeffs_unloaded)[None, :], None)
d_unloaded = jnp.zeros((1, 6)).at[0, 0].set(l)
eps_unloaded = struct.make_eps(d_unloaded)
assert jnp.allclose(eps_unloaded, 0.0), "Axial strain calculation incorrect"
g_int_unloaded, _ = struct.make_g_int_and_k_t(d_unloaded)
assert jnp.allclose(g_int_unloaded, 0.0), "Internal force vector should be zero for unloaded structure"

# check axial strain case - working
k_coeffs_axial = jnp.full(6, 1e5).at[0].set(l)
struct.set_design_variables(coords, jnp.diag(k_coeffs_axial)[None, :], None)
dx = 0.1
d_axial = jnp.zeros((1, 6))
d_axial = d_axial.at[0, 0].set(l + dx)
eps_axial = struct.make_eps(d_axial)
assert jnp.allclose(eps_axial, jnp.array((dx / l, 0.0, 0.0, 0.0, 0.0, 0.0))), "Axial strain calculation incorrect"
g_int_axial, _ = struct.make_g_int_and_k_t(d_axial)
expected_f_axial = jnp.zeros(12)
expected_f_axial = expected_f_axial.at[0].set(-k_coeffs_axial[0] * dx / l)
expected_f_axial = expected_f_axial.at[6].set(k_coeffs_axial[0] * dx / l)
assert jnp.allclose(g_int_axial, expected_f_axial), "Axial force calculation incorrect"

# check bending in y case - working
k_coeffs_bending = jnp.full(6, 3e5).at[4].set(1.0)
struct.set_design_variables(coords, jnp.diag(k_coeffs_bending)[None, :], None)
curvature_y = 0.1
d_bending_y = jnp.zeros((1, 6))
d_bending_y = d_bending_y.at[0, 0].set(l)
d_bending_y = d_bending_y.at[0, 4].set(curvature_y * l)
eps_bending_y = struct.make_eps(d_bending_y)
expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, curvature_y, 0.0))
assert jnp.allclose(eps_bending_y, expected_bending_strain), "Bending strain calculation incorrect"
g_ind_bending_y, _ = struct.make_g_int_and_k_t(d_bending_y)
expected_f_bending_y = jnp.zeros(12)
expected_f_bending_y = expected_f_bending_y.at[4].set(-k_coeffs_bending[4] * curvature_y)
expected_f_bending_y = expected_f_bending_y.at[10].set(k_coeffs_bending[4] * curvature_y)
assert jnp.allclose(g_ind_bending_y, expected_f_bending_y), "Bending moment calculation incorrect"

# check bending in z case - working
k_coeffs_bending = jnp.full(6, 3e5).at[5].set(1.0)
struct.set_design_variables(coords, jnp.diag(k_coeffs_bending)[None, :], None)
curvature_z = 0.1
d_bending_z = jnp.zeros((1, 6))
d_bending_z = d_bending_z.at[0, 0].set(l)
d_bending_z = d_bending_z.at[0, 5].set(curvature_z * l)
eps_bending_z = struct.make_eps(d_bending_z)
expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, curvature_z))
assert jnp.allclose(eps_bending_z, expected_bending_strain), "Bending strain calculation incorrect"
g_ind_bending_z, _ = struct.make_g_int_and_k_t(d_bending_z)
expected_f_bending_z = jnp.zeros(12)
expected_f_bending_z = expected_f_bending_z.at[5].set(-k_coeffs_bending[5] * curvature_z)
expected_f_bending_z = expected_f_bending_z.at[11].set(k_coeffs_bending[5] * curvature_z)
assert jnp.allclose(g_ind_bending_z, expected_f_bending_z), "Bending moment calculation incorrect"

# check shear in y case - does not work
# k_coeffs_shear_y = jnp.full(6, 3e5).at[1].set(1.0)
# struct.set_design_variables(coords, jnp.diag(k_coeffs_shear_y)[None, :], None)
# dy = 0.1
# d_shear_y = jnp.zeros((1, 6))
# d_shear_y = d_shear_y.at[0, 0].set(l)
# d_shear_y = d_shear_y.at[0, 1].set(dy)
# eps_shear_y = struct.make_eps(d_shear_y)
# expected_shear_strain = jnp.array((0.0, dy / l, 0.0, 0.0, 0.0, 0.0))
# assert jnp.allclose(eps_shear_y, expected_shear_strain), "shear strain calculation incorrect"
# g_ind_shear_y, _ = struct.make_g_int_and_k_t(d_shear_y)
# expected_f_shear_y = jnp.zeros(12)
# expected_f_shear_y = expected_f_shear_y.at[1].set(-k_coeffs_shear_y[1] * dy / l)
# expected_f_shear_y = expected_f_shear_y.at[7].set(k_coeffs_shear_y[1] * dy / l)
# assert jnp.allclose(g_ind_shear_y, expected_f_shear_y), "shear force calculation incorrect"


pass
