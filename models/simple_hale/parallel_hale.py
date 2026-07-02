from copy import deepcopy
import time
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    r"""
    Case to demonstrate parallelisation of the simple_hale model using jax.pmap. The case runs a parameter sweep over 
    gust length and gust amplitude, and plots the root strains for each case. This takes ~6 seconds per dynamic case on 
    a MacBook Air M2 for parallelisation over 32 cases.
    """

    # number of cases for sweep
    n_gust_length: int = 32
    n_gust_amplitude: int = 1
    n_case: int = n_gust_length * n_gust_amplitude

    # constants for all cases
    u_inf_mag: float = 10.0  # free stream velocity magnitude
    physical_time: float = 10.0

    # Set the number of threads for JAX to use
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={n_case} --xla_cpu_multi_thread_eigen=true"
    ).strip()

    # jax imports must be done after setting the environment variable
    import jax
    from jax import numpy as jnp

    from aegrad.aero.flowfields import Constant, OneMinusCosine
    from aegrad.coupled import CoupledAeroelastic
    from models.simple_hale.simple_hale import generate_simple_hale

    jax.config.update("jax_enable_x64", True)

    assert jax.device_count() == n_case, (
        f"Expected {n_case} devices, got {jax.device_count()}"
    )

    # parameter sweep definition
    gust_lengths = jnp.linspace(0.1 * u_inf_mag, 5.0 * u_inf_mag, n_gust_length)
    gust_amplitudes = (
        jnp.linspace(0.0, 0.5, n_gust_amplitude) if n_gust_amplitude > 1 else [0.3]
    )

    # create trimmed case, which can be used to initialise all gust cases
    const_flowfield = Constant(
        u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
        rho=1.225,
        relative_motion=True,
    )

    base_hale = generate_simple_hale(flowfield=const_flowfield, sigma_wing=1.5)
    n_tstep = int(physical_time / float(base_hale.aero.dt)) + 1

    # trim the aircraft
    static_sol, trim_vars = base_hale.trim(
        prescribed_dofs=jnp.arange(6),
        zero_force_dofs=(0, 2, 4),  # balance drag, lift, and pitching moment
        trim_cs="elevator",
        thrust_nodes="thrust",
        trim_orientation="y",
        horseshoe=False,
    )

    # change from static aircraft/dynamic freestream to dynamic aircraft/static freestream
    dynamic_init = base_hale.initialise_dynamic(static_case=static_sol)

    # create objects for each gust case, and stack them into a single object for parallelisation
    cases = []
    for gust_length in gust_lengths:
        for gust_amplitude in gust_amplitudes:
            gust_flowfield = OneMinusCosine(
                u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
                rho=1.225,
                relative_motion=False,
                gust_length=gust_length,
                gust_amplitude=gust_amplitude * u_inf_mag,
                gust_x0=jnp.array((-2.0 * gust_length - 10.0, 0.0, 0.0)),
            )
            new_case = deepcopy(base_hale)
            new_case.aero.flowfield = gust_flowfield
            cases.append(new_case)
    stacked_case = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *cases)

    # function we want to parallelise over the cases, which solves the dynamic problem and returns the root strains
    # the output can be changed to be any other quantity of interest, e.g. tip deflection, rigid body motions, etc.
    def solve(hale_: CoupledAeroelastic):
        dynamic_sol = hale_.dynamic_solve(
            init_case=dynamic_init, prescribed_dofs=None, n_tstep=n_tstep
        )
        return dynamic_sol.structure.eps[
            :, 0, 3:
        ]  # root strains (element 0, rotational strains)

    # parallelise case and time - this includes compilation time
    t_start = time.time()
    root_strains = jax.pmap(solve)(stacked_case)
    jax.block_until_ready(root_strains)
    t_end = time.time()
    print(f"Time taken for {n_case} cases: {t_end - t_start:.2f} seconds")
    print(f"Time per case: {(t_end - t_start) / n_case:.2f} seconds")

    # plot strains
    t = jnp.arange(n_tstep) * base_hale.aero.dt
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(gust_lengths[0]), vmax=float(gust_lengths[-1]))
    for i_dir in range(3):
        fig, ax = plt.subplots()
        for i in range(n_case):
            ax.plot(
                t,
                root_strains[i, :, i_dir],
                color=cmap(i / max(n_case - 1, 1)),
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Root Strain [m/m]")
        ax.set_title(
            f"{['Torsional', 'In-plane bending', 'Out-of-plane bending'][i_dir]} strain at root"
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label="Gust length [m]")
        plt.show()
