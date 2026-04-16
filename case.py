import pickle
from typing import Optional

from jax import numpy as jnp
from jax import Array
import jax

from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.flowfields import OneMinusCosine
from aegrad.aero.utils import make_rectangular_grid
from aegrad.aero.uvlm import UVLM
from aegrad.coupled import CoupledAeroelastic
from aegrad.coupled.data_structures import AeroelasticFullStates, AeroelasticDesignVariables

from structure import BeamStructure

jax.config.update("jax_enable_x64", True)

# discretization
# m = 10
# n = 30
# m_star = 20
# n_tstep = 1000

# m = 5
# n = 10
# m_star = 8
# n_tstep = 10

m = 2
n = 5
m_star = 6
n_tstep = 10

c_ref = 1.0
b_ref = 6.0
u_inf = jnp.array((20.0, 0.0, 2.0))
u_inf_mag = jnp.linalg.norm(u_inf)
k_cs = jnp.diag(jnp.array((1e7, 1e7, 1e7, 1e5, 1e7, 2e4)))
m_cs = jnp.diag(jnp.array((10.0, 10.0, 10.0, 10.0, 10.0, 10.0)))
dt = c_ref / (m * u_inf_mag)

delta = dt * u_inf_mag
delta_w = delta * jnp.logspace(0.0, 1.0, m_star)

# beam non-design variables
y_vector = jnp.array((0.0, 0.0, 1.0))
conn = jnp.zeros((n, 2), dtype=int)
conn = conn.at[:, 0].set(jnp.arange(n))
conn = conn.at[:, 1].set(jnp.arange(1, n + 1))
beam = BeamStructure(num_nodes=n + 1, connectivity=conn, y_vector=y_vector)

# aero non-design variables
gd = GridDiscretization(m=m, n=n, m_star=m_star)
uvlm = UVLM(
    grid_shapes=[gd],
    dof_mapping=jnp.arange(n + 1),
    mirror_point=jnp.zeros(3),
    mirror_normal=jnp.array((0.0, 1.0, 0.0)),
)

wing = CoupledAeroelastic(beam, uvlm)

beam_coords = jnp.zeros((n + 1, 3)).at[:, 1].set(jnp.linspace(0, b_ref, n + 1))
grid = make_rectangular_grid(m, n, c_ref, ea=0.2)
dt = c_ref / (jnp.linalg.norm(u_inf) * m)
flowfield = OneMinusCosine(u_inf=u_inf, rho=1.225, relative_motion=True, gust_length=10.0, gust_amplitude=4.0,
                           gust_x0=jnp.array((-25.0, 0.0, 0.0)))

wing.set_design_variables(
    coords=beam_coords,
    k_cs=k_cs,
    m_cs=m_cs,
    m_lumped=None,
    dt=dt,
    flowfield=flowfield,
    delta_w=delta_w,
    x0_aero=grid,
)

# set tolerance to zero, rather than none, to prevent error messages
# wing.structure.struct_convergence_settings = ConvergenceSettings(max_n_iter=100, rel_disp_tol=0.0,
#                                                                  abs_disp_tol=0.0,
#                                                                  rel_force_tol=0.0, abs_force_tol=0.0)
# wing.fsi_convergence_settings = ConvergenceSettings(max_n_iter=40, rel_disp_tol=0.0, abs_disp_tol=0.0,
#                                                     rel_force_tol=0.0, abs_force_tol=0.0)

static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))

dynamic_sol = wing.dynamic_solve(init_case=static_sol, prescribed_dofs=jnp.arange(6), spectral_radius=1.0,
                                 gamma_dot_relaxation_factor=0.7,
                                 free_wake=False, dt=dt, n_tstep=n_tstep, include_unsteady_aero_force=True)

# obtain the index with the largest WRBM
i_max: int = int(jnp.argmax(dynamic_sol.structure.f_elem[:, 0, 5]))

print(i_max)


# objective which refers to a single timestep
def objective(states: AeroelasticFullStates, dv: AeroelasticDesignVariables,
              i_ts: Optional[int | Array]) -> Array:
    return jax.lax.select(
        i_ts == i_max,
        states.structure.f_elem[0, 5],
        jnp.zeros(()),
    )


dynamic_sol.plot(directory=f"./test_outputs/dynamic_coupled_adjoint")

static_grad, static_adj = wing.static_adjoint(case=static_sol, objective=objective, forward_adjoint=True)

dynamic_grad, dynamic_adj = wing.dynamic_adjoint(case=dynamic_sol, objective=objective,
                                                 p_varphi_p_x=-static_adj)

dynamic_grad.plot(case=dynamic_sol, directory=f"./test_outputs/dynamic_grad", i_ts=0)

jax.block_until_ready(dynamic_grad)
jax.block_until_ready(static_grad)
jax.block_until_ready(dynamic_adj)
jax.block_until_ready(static_adj)

out_dict = {
    "wing": wing,
    "static_sol": static_sol,
    "static_grad": static_grad,
    "static_adj": static_adj,
    "dynamic_sol": dynamic_sol,
    "dynamic_grad": dynamic_grad,
    "dynamic_adj": dynamic_adj,
}

with open("./test_outputs/cantilever_wing_output.pickle", "wb") as f:
    pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
