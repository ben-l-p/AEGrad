import jax
from jax import Array

from aegrad.aero.uvlm_utils import make_rectangular_grid
from aegrad.aero.data_structures import GridDiscretization, InputUnflattened
from aegrad.algebra.array_utils import ArrayList
from aegrad.algebra.linear_operators import LinearSystem, LinearOperator
from aegrad.aero.linear import LinearWakeType
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as rot
from aegrad.aero.case import AeroCase
from aegrad.aero.flowfields import Constant
from aegrad.print_output import set_verbosity, VerbosityLevel
from pathlib import Path

jax.disable_jit()

# gradient of operations w.r.t. chord length

set_verbosity(VerbosityLevel.NORMAL)

jax.debug_nans()

u_inf = jnp.array((10.0, 0.0, 1.0))
rho_inf = 2.5
m = 2
n = 2
m_star = 5
c_ref0 = jnp.array(1.0)
b_ref = 1.0
alpha = jnp.deg2rad(0.0)
ea = 0.0
physical_time = 4.0  # seconds

flowfield = Constant(u_inf, rho_inf, True)
dt = c_ref0 / (m * flowfield.u_inf_mag)
n_tstep = int(jnp.ceil(physical_time / dt))
disc = GridDiscretization(m, n, m_star)

beam_coords = jnp.zeros((n + 1, 3))
beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
rmat = rot.from_euler("xyz", jnp.array((0.0, alpha, 0.0))).as_matrix()

# static position
hg = jnp.zeros((n + 1, 4, 4))
hg = hg.at[:, 3, 3].set(1.0)
hg = hg.at[:, :3, :3].set(rmat[None, :, :])
hg = hg.at[:, :3, 3].set(beam_coords)

case = AeroCase(n_tstep, disc, False, jnp.arange(0, n + 1))

def func(c_ref: Array) -> LinearOperator:
    x_grid = make_rectangular_grid(m, n, c_ref, ea)

    # nonlinear case
    case.set_design_variables(dt, flowfield, None, x_grid, hg)
    case.solve_static()

    # linear case
    linear_model = case.linearise(0,
                  LinearWakeType.PRESCRIBED,
                  bound_upwash=False,
                  wake_upwash=False,
                  unsteady_force=True)
    return linear_model.sys.a.matrix

a_x = func(c_ref0)

grad_a_x = jax.jacobian(func)(c_ref0)

pass
