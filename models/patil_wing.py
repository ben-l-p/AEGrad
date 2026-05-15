from pathlib import Path

import jax
from jax import numpy as jnp
from jax import Array

from aegrad.coupled import CoupledAeroelastic
from aegrad.structure import BeamStructure
from aegrad.aero.uvlm import UVLM
from aegrad.aero.utils import make_rectangular_grid
from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.flowfields import Constant
from aero.utils import add_control_surface
from algebra.array_utils import ArrayList


def make_patil_wing(
    sigma: float = 1.0,
    n_nodes: int = 49,
    m: int = 4,
    m_star: int = 4 * 20,
    u_inf=jnp.array((30.0, 0.0, 0.0)),
    rho=0.088,
) -> CoupledAeroelastic:
    r"""
    Create the wing from “Nonlinear Aeroelasticity and Flight Dynamics of High-Altitude Long-Endurance Aircraft",
    Patil, M. J., Hodges, D. H., and Cesnik, C. E. S. This wing is used for comparison with "Optimal Rolling Maneuvers
    with Very Flexible Wings", Salvatore Maraniello and Rafael Palacios.
    """

    # constants
    b_ref = 16.0
    c_ref = 1.0
    ea = 0.5

    m_cs = jnp.diag(jnp.array((0.75, 0.75, 0.75, 0.1, 0.1, 0.1)))
    k_cs = jnp.diag(
        jnp.array(
            (
                1e7,
                1e7,
                1e7,
                sigma * 1e4,
                5e6,
                sigma * 2e4,
            )
        )
    )

    # beam non-design variables
    n_elem = n_nodes - 1
    n = n_nodes - 1
    y_vector = jnp.array((0.0, 0.0, 1.0))  # TODO: this should include aoa
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_nodes))
    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=y_vector,
        gravity=jnp.array((0.0, 0.0, -9.81)),
    )

    # create aero grid function with control surface
    def aero_grid_func(
        x0: ArrayList,
        *,
        left_aileron: Array,
        right_aileron: Array,
    ) -> ArrayList:
        # positive is surface up for both left and right
        arr = add_control_surface(
            grid=x0[0],
            angle=left_aileron,
            m_slice=slice(int(0.75 * m), None),
            n_slice=slice(0, n // 4 + 1),
        )

        arr = add_control_surface(
            grid=arr,
            angle=right_aileron,
            m_slice=slice(int(0.75 * m), None),
            n_slice=slice(-n // 4 - 1, None),
        )

        return ArrayList([arr])

    # aero non-design variables
    gd = GridDiscretization(m=m, n=n, m_star=m_star)
    uvlm = UVLM(
        grid_shapes=[gd],
        dof_mapping=jnp.arange(n_nodes),
        grid_func=aero_grid_func,
    )

    wing = CoupledAeroelastic(beam, uvlm)

    beam_coords = (
        jnp.zeros((n_nodes, 3)).at[:, 1].set(jnp.linspace(-b_ref, b_ref, n_nodes))
    )
    grid = make_rectangular_grid(m, n, c_ref, ea)
    dt = c_ref / (jnp.linalg.norm(u_inf) * m)
    flowfield = Constant(u_inf=u_inf, rho=rho, relative_motion=True)

    wing.set_design_variables(
        coords=beam_coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=None,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=grid,
        reference_cs_angles={
            "left_aileron": jnp.zeros(()),
            "right_aileron": jnp.zeros(()),
        },
    )

    return wing


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    coupled_system = make_patil_wing(u_inf=jnp.array((30.0, 0.0, 1.0)))

    prescribed_dofs_static = jnp.arange(6) + (coupled_system.structure.n_nodes - 1) * 3

    prescribed_dofs_dynamic = (
        jnp.array((0, 1, 2, 4, 5)) + (coupled_system.structure.n_nodes - 1) * 3
    )

    static_sol = coupled_system.static_solve(prescribed_dofs=prescribed_dofs_static)

    n_tstep = 200

    dynamic_sol = coupled_system.dynamic_solve(
        init_case=static_sol,
        prescribed_dofs=prescribed_dofs_dynamic,
        n_tstep=n_tstep,
        cs_ang_t={
            "left_aileron": jnp.linspace(0.0, -jnp.deg2rad(30.0), n_tstep),
            "right_aileron": jnp.linspace(0.0, -jnp.deg2rad(30.0), n_tstep),
        },
    )

    dir_ = Path("./coupled_patil")
    dynamic_sol.plot(dir_)
