from __future__ import annotations
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
from jax import Array

from aegrad.algebra.array_utils import ArrayList
from aegrad.aero.data_structures import GridDiscretization
from aegrad.aero.uvlm import UVLM
from aegrad.aero.linear import LinearWakeType, InputUnflattened
from aegrad.aero.utils import biot_savart_cutoff, make_rectangular_grid
from aegrad.aero.flowfields import Constant

if __name__ == "__main__":
    r"""
    Comparison for linear and nonlinear UVLM using a harmonic heaving wing case with two free ends. Saves the lift force
    time history to file for both cases for comparison.
    """
    # amplitude of oscillation, m
    ampl = 1.0  # small amplitude case

    u_inf: Array = jnp.array((10.0, 0.0, 1.0))  # freestream velocity
    rho_inf: float = 1.225  # freestream density
    m = 10  # number of chordwise panels
    n = 20  # number of spanwise panels
    m_star = m * 5  # number of wake panels
    physical_time = 5.0  # seconds

    c_ref = 1.0  # chord length
    b_ref = 5.0  # wing span
    alpha = jnp.deg2rad(0.0)  # wing angle of incidence

    dt = c_ref / (m * jnp.linalg.norm(u_inf))  # time step length, s
    n_tstep = int(jnp.ceil(physical_time / dt))  # number of time steps for simulation
    disc = GridDiscretization(
        m, n, m_star
    )  # object to store surface aerodynamic discretisation

    x_grid = make_rectangular_grid(
        m, n, c_ref, 0.4
    )  # aerodynamic grid local coordinates

    # beam coordinates using a cosine spanwise discretisation
    beam_coords = jnp.zeros((n + 1, 3))
    beam_coords = beam_coords.at[:, 1].set(
        -0.5 * b_ref * jnp.cos(jnp.linspace(0.0, jnp.pi, n + 1)) + 0.5 * b_ref
    )
    rmat = Rot.from_euler("xyz", jnp.array((0.0, alpha, 0.0))).as_matrix()

    # reference SE(3) coordinates
    hg = jnp.zeros((n + 1, 4, 4))
    hg = hg.at[:, 3, 3].set(1.0)
    hg = hg.at[:, :3, :3].set(rmat[None, :, :])
    hg = hg.at[:, :3, 3].set(beam_coords)

    # wake discretisation
    delta = dt * jnp.linalg.norm(u_inf)
    delta_w = delta * jnp.logspace(0.0, 0.7, m_star)

    # nonlinear case
    nonlinear_model = UVLM(
        grid_shapes=[disc],
        dof_mapping=jnp.arange(0, n + 1),
        kernel=biot_savart_cutoff,
        gamma_dot_relaxation=1.0,
    )
    nonlinear_model.set_design_variables(
        dt=dt,
        flowfield=Constant(u_inf=u_inf, rho=rho_inf, relative_motion=True),
        x0_b=x_grid,
        hg0=hg,
        delta_w=delta_w,
    )
    # solve static case
    case = nonlinear_model.solve_static()

    # heaving motion
    freq = 6.0  # Hz
    omega = 0.5 * jnp.pi * freq  # rad/s
    k = freq * c_ref / (2.0 * u_inf[0])
    print(f"Frequency: {freq:.2f} Hz, Reduced Frequency: {float(k):.2f}")

    t = jnp.arange(n_tstep) * dt
    z_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
    z_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

    hg_t = jnp.zeros((n_tstep, n + 1, 4, 4))
    hg_t = hg_t.at[:, 3, 3].set(1.0)
    hg_t = hg_t.at[...].set(hg[None, ...])
    hg_t = hg_t.at[:, :, 2, 3].add(z_t[:, None])

    hg_dot_t = jnp.zeros_like(hg_t)
    hg_dot_t = hg_dot_t.at[:, :, 2, 3].set(z_dot_t[:, None])

    static_case = nonlinear_model.solve_static()

    # nonlinear case
    nonlinear_case = nonlinear_model.solve_prescribed_dynamic(
        init_case=static_case,
        hg_t=hg_t,
        hg_dot_t=hg_dot_t,
    )

    # save nonlinear wing time history
    nonlinear_case.plot(Path(f"./test_outputs/heaving_test_nonlinear_ampl_{ampl:.02f}"))

    # linear case
    linear_model = nonlinear_model.linearise(
        reference=static_case,
        # wake_type=LinearWakeType.PRESCRIBED,
        wake_type=LinearWakeType.FROZEN,
        bound_upwash=False,
        wake_upwash=False,
        unsteady_force=True,
    )

    # use nonlinear grid coordinates as input for the linear case
    delta_zeta_b = nonlinear_case.zeta_b - ArrayList(
        [zeta[None, ...] for zeta in linear_model.reference.zeta_b]
    )
    u_linear = InputUnflattened(
        zeta_b=delta_zeta_b,
        zeta_b_dot=nonlinear_case.zeta_b_dot,
        nu_b=None,
        nu_w=None,
    )

    # run linear case
    linear_case = linear_model.run(u_linear, use_matrix=True)

    # save linear wing time history
    linear_case.plot(Path(f"./test_outputs/heaving_test_linear_ampl_{ampl:.02f}"))

    # extract and save lift time history for both cases
    f_z_linear = (linear_case.y_t_tot.f_steady + linear_case.y_t_tot.f_unsteady)[0].sum(
        axis=(1, 2)
    )[:, 2]

    f_z_nonlinear = (nonlinear_case.f_steady + nonlinear_case.f_unsteady)[0].sum(
        axis=(1, 2)
    )[:, 2]

    out = np.stack([t, f_z_linear, f_z_nonlinear], axis=-1)
    np.savetxt(
        f"heaving_wing_lift_ampl_{ampl:.02f}.dat",
        out,
        comments="",
        header="t linear nonlinear",
    )
