from aegrad.aero.uvlm_utils import make_rectangular_grid, GridDiscretization
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from aegrad.aero.case import AeroCase
from aegrad.aero.flowfields import Constant
from pathlib import Path

u_inf = jnp.array((10.0, 0.0, 0.0))
rho_inf = 1.225
m = 10
n = 20
m_star = 30
c_ref = 1.0
b_ref = 5.0
alpha = jnp.deg2rad(5.0)
ea = 0.3
dt = jnp.array(0.1)

disc = GridDiscretization(m, n, m_star)
flowfield = Constant(u_inf, rho_inf, True)
delta_w = c_ref / m * jnp.outer(jnp.arange(1, m_star + 1), u_inf / jnp.linalg.norm(u_inf))

x_grid = make_rectangular_grid(m, n, c_ref, ea)

beam_coords = jnp.zeros((n + 1, 3))
beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
rmat = R.from_euler('xyz', jnp.array((0.0, alpha, 0.0))).as_matrix()

hg = jnp.zeros((n + 1, 4, 4))
hg = hg.at[:, :3, :3].set(rmat[None, :, :])
hg = hg.at[:, :3, 3].set(beam_coords)

case = AeroCase(1, disc, False, jnp.arange(0, n + 1))

case.set_design_variables(dt, flowfield, delta_w, x_grid, hg)

plot_path = Path("./plots/")
plot_path.mkdir(parents=True, exist_ok=True)
case.plot_reference(plot_path)

case.solve_static(0, None, True)

case.plot(plot_path)

pass
