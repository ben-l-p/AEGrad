from __future__ import annotations
from aegrad.algebra.se3 import p, x_rmat_to_hg, hg_to_d, k_tg_entry, log_se3, exp_se3
from aegrad.algebra.base import chi
import numpy as np
from aegrad.structure.solvers import newton_raphson

import jax
from jax import numpy as jnp
from jax import Array

from matplotlib import pyplot as plt

jax.config.update('jax_enable_x64', True)

n_node: int = 2
l_beam: float = 1.0

# coordinates
x0 = jnp.zeros((n_node, 3)) # [n_node, 3]
x0 = x0.at[:, 0].set(jnp.linspace(0, l_beam, n_node))

# stiffness
eiz = 5.0
mz = 1e-4
k_cs = jnp.diag(jnp.array((100.0, 100.0, 100.0, 100.0, 100.0, eiz)))    # [6, 6]

# connectivity
n_elem: int = n_node - 1
conn = jnp.zeros((n_elem, 2), dtype=int)    # [n_elem, 2]
conn = conn.at[:, 0].set(jnp.arange(0, n_elem))
conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

# dofs per element
dof_per_elem = jnp.zeros((n_elem, 12), dtype=int)
for i_elem in range(n_elem):
    i_node1, i_node2 = conn[i_elem, :]
    dof_per_elem = dof_per_elem.at[i_elem, :6].set(jnp.arange(i_node1 * 6, i_node1 * 6 + 6))
    dof_per_elem = dof_per_elem.at[i_elem, 6:].set(
        jnp.arange(i_node2 * 6, i_node2 * 6 + 6)
    )

# external follower forces
g_ext = jnp.zeros((n_node, 6))
g_ext = g_ext.at[-1, 5].set(mz)   # [n_node, 6]

# define coordinate system
# this is a vector per undeformed element which points in the planform normal direction
# normalisation would be added here for implementation
u_vec = jnp.zeros((n_elem, 3))  # [n_elem, 3]
u_vec = u_vec.at[:, 2].set(1.0)

d_x0 = jnp.take(x0, conn[:, 1], axis=0) - jnp.take(x0, conn[:, 0], axis=0)  # undeformed segment vectors, [n_elem, 3]
l_x0 = jnp.linalg.norm(d_x0, axis=1)  # undeformed segment lengths, [n_elem]
d_vec = d_x0 / l_x0[:, None]   # undeformed unit segment vectors, [n_elem, 3]

t_vec = jnp.cross(u_vec, d_vec, axis=-1)
u_vec_ = jnp.cross(d_vec, t_vec, axis=-1)

o0 = jnp.stack((d_vec, t_vec, u_vec_), axis=-1)     # [n_elem, 3, 3]

# baseline deformation gradient
f0 = jnp.zeros((n_elem, 6))     # [n_elem, 6]
f0 = f0.at[:, :3].set(d_vec)

# stiffness matrix
k = jnp.zeros((n_elem, 6, 6))   # [n_elem, 6, 6]

for i_elem in range(n_elem):
    this_o0 = o0[i_elem, ...]   # [3, 3]
    chi_ = chi(this_o0)     # [6, 6]
    k = k.at[i_elem, ...].set(chi_ @ k_cs @ chi_.T) # [6, 6]

# tangent stiffness
n_dof: int = n_node * 6
def make_k_tm(d: Array) -> Array:
    k_tm = jnp.zeros((n_dof, n_dof))

    pd = jax.vmap(p, 0, 0)(d)   # [n_elem, 6, 12]
    pd_l = pd / l_x0[:, None, None]  # [n_elem, 6, 12]

    k_tm_entries = jnp.einsum('ikj,ikl,ilm->ijm', pd, k, pd_l)

    for i_elem in range(n_elem):
        index = dof_per_elem[i_elem, :]
        k_tm = k_tm.at[jnp.ix_(index, index)].set(k_tm_entries[i_elem, ...])
    return k_tm

# geometric stiffness
def make_k_tg(d: Array) -> Array:
    k_tg = jnp.zeros((n_dof, n_dof))

    eps = (d - d0) / l_x0

    k_tg_entries = jax.vmap(k_tg_entry, (0, 0, 0), 0)(d, eps, k)   # [n_elem, 12, 12]

    for i_elem in range(n_elem):
        index = dof_per_elem[i_elem, :]
        k_tg = k_tg.at[jnp.ix_(index, index)].set(k_tg_entries[i_elem, ...])
    return k_tg

# baseline test
hg0 = jax.vmap(lambda x_: x_rmat_to_hg(x_, jnp.eye(3)), 0, 0)(x0)   # [n_node, 4, 4]
ha0 = jax.vmap(lambda hg_: log_se3(hg_), 0, 0)(hg0)    # [n_node, 6]
d0 = jax.vmap(hg_to_d, (0, 0), 0)(hg0[:-1, ...], hg0[1:, ...])

test_k_tm = np.array(make_k_tm(d0))
test_k_tg = np.array(make_k_tg(d0))

def make_g_int(d: Array) -> Array:
    eps = (d - d0) / l_x0

    pd = jax.vmap(p, 0, 0)(d)  # [n_elem, 6, 12]

    g_int_full = jnp.einsum('ijk,ijl,il->ik', pd, k, eps)  # [n_elem, 12]

    g_int = jnp.zeros(n_dof)
    for i_elem in range(n_elem):
        g_int = g_int.at[dof_per_elem[i_elem, :]].add(g_int_full[i_elem, :])
    return g_int

def func(ha: Array) -> Array:
    ha_mat = ha.reshape(n_node, 6)
    hg_mat = jax.vmap(exp_se3, 0, 0)(ha_mat)   # [n_node, 4, 4]
    d = jax.vmap(hg_to_d, (0, 0), 0)(hg_mat[:-1, ...], hg_mat[1:, ...])
    return make_g_int(d) - g_ext.ravel()

def jac(ha: Array) -> Array:
    ha_mat = ha.reshape(n_node, 6)
    hg_mat = jax.vmap(exp_se3, 0, 0)(ha_mat)  # [n_node, 4, 4]
    d = jax.vmap(hg_to_d, (0, 0), 0)(hg_mat[:-1, ...], hg_mat[1:, ...])
    return make_k_tm(d) + make_k_tg(d)

free_dof = slice(6, None)
h_sol = newton_raphson(func, jac, ha0.ravel(), free_dof)
hg_sol = jax.vmap(exp_se3, 0, 0)(h_sol.reshape(n_node, 6))   # [n_node, 4, 4]
d_sol = jax.vmap(hg_to_d, (0, 0), 0)(hg_sol[:-1, ...], hg_sol[1:, ...]) # [n_elem, 6]

# interpolated coordinates
s = jnp.linspace(0.0, 1.0, 10) / l_x0[0]

delta = jax.vmap(lambda i_elem_: jax.vmap(lambda s_l_: exp_se3(d_sol[i_elem_, :] * s_l_), 0, 0)(s))(jnp.arange(n_elem))  # [n_elem, n_interp, 4, 4]
hg_interp = jnp.einsum('ijk,ilkm->iljm', hg_sol[:-1, ...], delta)   # [n_elem, n_interp, 4, 4]
hg_interp_line = hg_interp.reshape(-1, 4, 4)   # [n, 4, 4]

x_sol = hg_interp_line[:, :3, 3]  # [n_node, 3]
rmat_sol = hg_interp_line[:, :3, :3]  # [n_node, 3, 3]

fig, ax = plt.subplots()
ax.plot(x_sol[:, 0], x_sol[:, 1])
ax.quiver(x_sol[:, 0], x_sol[:, 1], rmat_sol[:, 0, 0], rmat_sol[:, 1, 0], color='r')
ax.quiver(x_sol[:, 0], x_sol[:, 1], rmat_sol[:, 0, 1], rmat_sol[:, 1, 1], color='b')
ax.axis("equal")
fig.show()

# resultant forces
g_int = make_g_int(d_sol).reshape(-1, 6)

mz_root, mz_tip = g_int[:, 5]

m_result_tip = mz_tip - mz

kz = d_sol[:, 5]
expected_kz = mz / eiz

pass
