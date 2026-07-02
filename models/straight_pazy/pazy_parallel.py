if __name__ == "__main__":
    r"""
    Run a grid of Pazy cases with varying AoA and dynamic pressure in parallel using JAX's pmap. Using the provided
    discretization, can run 1024 cases in under a minute on an M2 MacBook Air. Note that imports are done inside the 
    main function to avoid JAX initializing devices before setting the XLA flags. The results are plotted as a heatmap 
    and line plots for each AoA.
    """

    import os

    n_q = 32
    n_alpha = 32
    n_case = n_q * n_alpha
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={n_case} --xla_cpu_multi_thread_eigen=true"
    ).strip()

    import time
    from matplotlib import pyplot as plt

    import jax
    from jax import numpy as jnp

    from aegrad.aero.flowfields import Constant
    from aegrad.coupled import CoupledAeroelastic
    from aegrad.utils.print_utils import VerbosityLevel, set_verbosity
    from models.straight_pazy.pazy_wing import make_pazy_wing

    jax.config.update("jax_enable_x64", True)
    set_verbosity(VerbosityLevel.SILENT)

    assert jax.device_count() == n_case, (
        f"Expected {n_case} devices, got {jax.device_count()}"
    )

    rho = 1.225
    q_inf_vec = jnp.linspace(1.0, 3000.0, n_q)
    alphas = jnp.deg2rad(jnp.linspace(-5.0, 15.0, n_alpha))

    cases = []
    for q_inf in q_inf_vec:
        u_inf_mag = jnp.sqrt(2 * q_inf / rho)
        for alpha in alphas:
            cases.append(
                make_pazy_wing(
                    flowfield=Constant(
                        u_inf=jnp.array(
                            (
                                u_inf_mag,
                                0.0,
                                0.0,
                            )
                        ),
                        rho=rho,
                        relative_motion=True,
                    ),
                    aoa=alpha,
                    m=12,
                    node_multiplier=2,
                )
            )

    stacked_case = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *cases)

    def solve(case_: CoupledAeroelastic):
        static_sol = case_.static_solve(
            prescribed_dofs=tuple(range(6)),
            horseshoe=True,
        )

        return (
            0.5
            * (
                static_sol.aero.zeta_b[0][0, -1, 2]
                + static_sol.aero.zeta_b[0][-1, -1, 2]
            )
            / 0.55
        )

    parallel_func = jax.pmap(solve)

    t_start = time.time()
    tip_z = parallel_func(stacked_case)
    jax.block_until_ready(tip_z)
    t_end = time.time()

    print("Total time: ", t_end - t_start)
    print("Time per case: ", (t_end - t_start) / n_case)

    tip_z_grid = tip_z.reshape(n_q, n_alpha).T

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        q_inf_vec, jnp.rad2deg(alphas), tip_z_grid, shading="auto", cmap="bwr"
    )
    fig.colorbar(mesh, ax=ax, label="Tip Z Displacement [z/b]")
    ax.set_xlabel("Dynamic Pressure [Pa]")
    ax.set_ylabel("Angle of Attack [deg]")
    ax.set_title("Pazy Wing Tip Z Displacement")
    plt.show()

    alphas_deg = jnp.rad2deg(alphas)
    norm = plt.Normalize(vmin=float(alphas_deg.min()), vmax=float(alphas_deg.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots()
    for i_aoa, aoa in enumerate(alphas_deg):
        ax.plot(q_inf_vec, tip_z_grid[i_aoa], color=cmap(norm(float(aoa))))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Angle of Attack [deg]")
    ax.set_xlabel("Dynamic Pressure [Pa]")
    ax.set_ylabel("Tip Z Displacement [z/b]")
    ax.set_title("Pazy Wing Tip Z Displacement")
    plt.show()
