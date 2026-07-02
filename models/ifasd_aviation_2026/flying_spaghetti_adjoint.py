from pathlib import Path
import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional

from models.flying_spaghetti import generate_flying_spaghetti

from aegrad.structure import StructureFullStates, StructuralDesignVariables
from aegrad.utils.data_structures import ConvergenceSettings

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    r"""
    Obtain the time history of the flying spaghetti case, and the gradient of the tip node x-coordinate with respect to
    the mass per unit length. This case verifies gradients with finite differences.
    """
    n_nodes_ = 21
    dt_ = 0.01
    t_end_ = 10.0
    n_tstep_ = int(jnp.ceil(t_end_ / dt_)) + 1

    # start one timestep behind to prevent issues with initial condition
    t_ = jnp.arange(n_tstep_) * dt_ - dt_

    def make_sol(k_cs_eps: float, m_cs_eps: float):
        struct_, f_dead_2d_, f_dead_3d_ = generate_flying_spaghetti(
            n_nodes_, t_, spectral_radius=0.7
        )
        struct_.k_cs = struct_.k_cs.at[0, 4, 4].add(k_cs_eps)
        struct_.m_cs = struct_.m_cs.at[0, :3, :3].add(
            jnp.diag(jnp.array((m_cs_eps, m_cs_eps, m_cs_eps)))
        )

        # convergence very strict, forces 100 structural iterations
        struct_.struct_convergence_settings = ConvergenceSettings(
            max_n_iter=100,
            rel_disp_tol=0.0,
            abs_disp_tol=0.0,
            rel_force_tol=0.0,
            abs_force_tol=0.0,
        )
        return (
            struct_,
            struct_.dynamic_solve(
                init_state=None,
                n_tstep=n_tstep_,
                dt=dt_,
                f_ext_follower=None,
                f_ext_dead=f_dead_2d_,  # swap between 2d and 3d to see the difference in response
                f_ext_aero=None,
                prescribed_dofs=None,
            ),
        )

    base_struct, base_sol = make_sol(k_cs_eps=0.0, m_cs_eps=0.0)

    i_ts_plot = jnp.linspace(0.0, n_tstep_ - 1, 6).astype(int).tolist()

    # data for plot
    xz_t = jnp.transpose(
        base_sol.hg[i_ts_plot, :, :, 3][:, :, (0, 2)],
        (1, 2, 0),
    )
    import numpy as np

    np.savetxt(
        "spaghetti_coords.dat",
        xz_t.reshape(n_nodes_, -1),
        comments="",
        header="x0 x1 x2 x3 x4 x5 z0 z1 z2 z3 z4 z5",
    )

    plot_path = Path("./flying_spaghetti_outputs/")
    stride = 10
    base_sol.plot(plot_path, n_interp=3, index=jnp.arange(0, n_tstep_, stride))

    # objective which refers to a single timestep
    def objective(
        states: StructureFullStates,
        _: StructuralDesignVariables,
        i_ts: Optional[int | Array],
    ) -> Array:
        return jax.lax.select(
            i_ts == n_tstep_ - 1, states.hg[-1, 0, 3], 0.0
        )  # tip coordinate

    grads, adj = base_struct.dynamic_adjoint(
        structure=base_sol, objective=objective, approx_grads=False
    )
    jax.block_until_ready(grads)
    jax.block_until_ready(adj)

    if grads.m_cs is None:
        raise ValueError("m_cs is None")
    ad_grad_m_cs = (
        grads.m_cs[0, 0, 0, 0] + grads.m_cs[0, 0, 1, 1] + grads.m_cs[0, 0, 2, 2]
    )

    # obtain gradients via finite differences
    pert_m_cs_obj = make_sol(k_cs_eps=0.0, m_cs_eps=1e-3)[1].hg[-1, -1, 0, 3]
    fd_grad_m_cs = (pert_m_cs_obj - base_sol.hg[-1, -1, 0, 3]) / 1e-3

    print(f"Adjoint grad_m_cs: {float(ad_grad_m_cs)}")
    print(f"Finite difference grad_m_cs: {float(fd_grad_m_cs)}")
