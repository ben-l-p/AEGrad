from aegrad.aero.uvlm_utils import make_rectangular_grid
from aegrad.aero.data_structures import GridDiscretization
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from aegrad.aero.case import AeroCase
from aegrad.aero.flowfields import Constant
import jax
from pathlib import Path
jax.config.update("jax_debug_nans", True)

u_inf = jnp.array((10.0, 0.0, 1.0))
rho_inf = 1.225
m = 2
n = 4
m_star = 8
c_ref = 1.0
b_ref = 1.0
alpha = jnp.deg2rad(0.0)
ea = 0.0

disc = GridDiscretization(m, n, m_star)
flowfield = Constant(u_inf, rho_inf, True)
dt = c_ref / (m * jnp.linalg.norm(u_inf))

x_grid = make_rectangular_grid(m, n, c_ref, ea)

beam_coords = jnp.zeros((n + 1, 3))
beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
rmat = R.from_euler('xyz', jnp.array((0.0, alpha, 0.0))).as_matrix()

hg = jnp.zeros((n + 1, 4, 4))
hg = hg.at[:, :3, :3].set(rmat[None, :, :])
hg = hg.at[:, :3, 3].set(beam_coords)

case = AeroCase(1, disc, False, jnp.arange(0, n + 1))

case.set_design_variables(dt, flowfield, None, x_grid, hg)

plot_path = Path("./plots/")
plot_path.mkdir(parents=True, exist_ok=True)

case.plot_reference(plot_path)

case.solve_static(0, None, True)

case.plot(plot_path, None)

case.linearise(0)

# case.plot(plot_path)

pass
