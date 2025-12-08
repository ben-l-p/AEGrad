from aegrad.aero.uvlm_utils import make_rectangular_grid
from aegrad.aero.data_structures import GridDiscretization, InputUnflattened
from aegrad.algebra.array_utils import ArrayList
from aegrad.aero.linear import LinearWakeType
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as rot
from aegrad.aero.case import AeroCase
from aegrad.aero.flowfields import Constant
from pathlib import Path

u_inf = jnp.array((10.0, 0.0, 0.0))
rho_inf = 1.225
m = 4
n = 8
m_star = 10
c_ref = 1.0
b_ref = 5.0
alpha = jnp.deg2rad(0.0)
ea = 0.0
physical_time = 6.0  # seconds

flowfield = Constant(u_inf, rho_inf, True)
dt = c_ref / (m * flowfield.u_inf_mag)
n_tstep = int(jnp.ceil(physical_time / dt))
disc = GridDiscretization(m, n, m_star)

x_grid = make_rectangular_grid(m, n, c_ref, ea)

beam_coords = jnp.zeros((n + 1, 3))
beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
rmat = rot.from_euler("xyz", jnp.array((0.0, alpha, 0.0))).as_matrix()

# static position
hg = jnp.zeros((n + 1, 4, 4))
hg = hg.at[:, 3, 3].set(1.0)
hg = hg.at[:, :3, :3].set(rmat[None, :, :])
hg = hg.at[:, :3, 3].set(beam_coords)

# heaving motion
freq = 3.0  # Hz
omega = 0.5 * jnp.pi * freq
ampl = 0.3  # m
t = jnp.arange(n_tstep) * dt
z_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
z_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

hg_t = jnp.zeros((n_tstep, n + 1, 4, 4))
hg_t = hg_t.at[:, 3, 3].set(1.0)
hg_t = hg_t.at[...].set(hg[None, ...])
hg_t = hg_t.at[:, :, 2, 3].add(z_t[:, None])

hg_dot_t = jnp.zeros_like(hg_t)
hg_dot_t = hg_dot_t.at[:, :, 2, 3].set(z_dot_t[:, None])

# nonlinear case
path_nl = Path("./plot_heaving_nl")
path_nl.mkdir(parents=True, exist_ok=True)
case = AeroCase(n_tstep, disc, False, jnp.arange(0, n + 1))
case.set_design_variables(dt, flowfield, None, x_grid, hg)
case.solve_static()
case.solve_prescribed_dynamic(hg_t, hg_dot_t, False)
case.plot(path_nl)

# linear case
case.surf_b_names = ["linear_surf_0"]
case.surf_w_names = ["linear_wake_0"]

path_lin = Path("./plot_heaving_lin")
path_lin.mkdir(parents=True, exist_ok=True)
linear_model = (case.
                linearise(0,
              LinearWakeType.PRESCRIBED,
              bound_upwash=False,
              wake_upwash=False,
              unsteady_force=True))

delta_zeta_b = case.zeta_b - ArrayList([zeta[None, ...] for zeta in linear_model.zeta0_b])
u_linear = InputUnflattened(zeta_b=delta_zeta_b,
                            zeta_b_dot=case.zeta_b_dot,
                            nu_b=None,
                            nu_w=None,
)

linear_model.run(u_linear)
linear_model.plot(path_lin)

pass