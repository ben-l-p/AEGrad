if __name__ == "__main__":
    r"""
    Perform a time-domain LCO computation for the straight Pazy wing, plotting the deflection of the tip of the beam
    over time.
    """
    import jax
    from jax import numpy as jnp

    from aegrad.aero.flowfields import Constant
    from models.straight_pazy.pazy_wing import make_pazy_wing

    from matplotlib import pyplot as plt

    jax.config.update("jax_enable_x64", True)

    u_inf_mag = 74.0
    alpha = jnp.deg2rad(0.5)

    f_tip = 0.0
    m_tip = 0.5
    physical_time = 1.5

    case = make_pazy_wing(
        flowfield=Constant(
            u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
            rho=1.225,
            relative_motion=True,
        ),
        aoa=alpha,
        m=10,
        m_star=100,
        node_multiplier=2,
    )

    dt = case.aero.dt
    n_tstep = int(physical_time / dt)
    print(
        f"u={u_inf_mag:.0f} m/s, alpha={float(jnp.rad2deg(alpha)):.1f} deg, dt={float(dt):.4f} s, n_tstep={n_tstep}"
    )

    f_ext = jnp.zeros((case.structure.n_nodes, 6)).at[-1, 2].set(f_tip)
    f_ext = f_ext.at[-1, 3].set(m_tip)

    static_sol = case.static_solve(
        prescribed_dofs=tuple(range(6)), horseshoe=False, f_ext_follower=f_ext
    )

    dynamic_sol = case.dynamic_solve(
        init_case=static_sol, prescribed_dofs=tuple(range(6)), n_tstep=n_tstep
    )

    dynamic_sol.plot("./out/")

    tip_z = dynamic_sol.structure.x[:, -1, 2]
    t = jnp.arange(n_tstep) * dt

    fig, ax = plt.subplots()
    ax.plot(t, tip_z)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Tip deflection, m")
    fig.show()
