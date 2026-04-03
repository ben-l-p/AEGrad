from pathlib import Path

import jax
from jax import numpy as jnp
from jax import Array

from coupled.gradients.coupled import CoupledAeroelastic
from structure import BeamStructure
from aero.uvlm import UVLM
from aero.utils import make_rectangular_grid
from aero.data_structures import GridDiscretization
from aero.flowfields import Constant

jax.config.update("jax_enable_x64", True)


def make_cantilever_wing(
        n_nodes: int = 40,
        b_ref: float = 5.0,
        c_ref: float = 1.0,
        ea: float = 0.25,
        m: int = 10,
        m_star: int = 20,
        k_cs: Array = jnp.diag(jnp.array((1e6, 1e6, 1e6, 4e2, 4e2, 4e2))),
        u_inf=jnp.array((10.0, 0.0, 1.0)),
        rho=1.225,
) -> CoupledAeroelastic:
    n_elem = n_nodes - 1
    n = n_nodes - 1

    # beam non-design variables
    y_vector = jnp.array((0.0, 0.0, 1.0))
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_nodes))
    beam = BeamStructure(num_nodes=n_nodes, connectivity=conn, y_vector=y_vector)

    # aero non-design variables
    gd = GridDiscretization(m=m, n=n, m_star=m_star)
    uvlm = UVLM(
        grid_shapes=[gd],
        dof_mapping=jnp.arange(n_nodes),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),
    )

    wing = CoupledAeroelastic(beam, uvlm)

    beam_coords = jnp.zeros((n_nodes, 3)).at[:, 1].set(jnp.linspace(0, b_ref, n_nodes))
    grid = make_rectangular_grid(m, n, c_ref, ea)
    dt = c_ref / (jnp.linalg.norm(u_inf) * m)
    flowfield = Constant(u_inf=u_inf, rho=rho, relative_motion=True)

    wing.set_design_variables(
        coords=beam_coords,
        k_cs=k_cs,
        m_cs=None,
        m_lumped=None,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=grid,
    )

    return wing


if __name__ == "__main__":
    coupled_system = make_cantilever_wing()
    result = coupled_system.static_solve(
        f_ext_dead=None,
        f_ext_follower=None,
        prescribed_dofs=jnp.arange(6),
    )

    dir_ = Path("./coupled_cantilever")
    result.plot(dir_)
