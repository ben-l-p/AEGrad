from typing import Optional, Literal

import jax
from jax import numpy as jnp
from jax import Array

from aegrad.aero.flowfields import FlowField, Constant
from aegrad.coupled import CoupledAeroelastic

from models.pazy.base import make_generic_pazy_wing
from models.pazy.straight.data.prepazy_properties import (
    PREPAZY_NO_SKIN,
    PREPAZY_WITH_SKIN,
)
from models.pazy.straight.data.technion_pazy_properties import (
    TECHNION_PAZY_NO_SKIN,
    TECHNION_PAZY_WITH_SKIN,
)


def make_pazy_wing(
    m: int = 12,
    m_star: int = 120,
    skin: bool = True,
    model: Literal["prepazy", "pazy"] = "pazy",
    node_multiplier: int = 1,
    gravity: bool | Array = False,
    flowfield: FlowField = Constant(
        u_inf=jnp.array((40.0, 0.0, 0.0)), rho=1.225, relative_motion=True
    ),
    aoa: float | Array = jnp.deg2rad(3.0),
    sweep: Optional[float | Array] = None,
    variable_disc_wake: bool = False,
    custom_dt: Optional[float] = None,
) -> CoupledAeroelastic:
    r"""
    Function to construct an instance of the Pazy wing model.
    :param m: Number of chordwise panels.
    :param m_star: Number of wake-wise panels.
    :param skin: Whether to use the structural model for skin on or skin off.
    :param model: Determine which Pazy variant to use, "Pazy" or "Prepazy" available.
    :param node_multiplier: Integer multiplier for the number of nodes in the structural model.
    :param gravity: Optional gravity vector. If None, no gravity is used.
    :param flowfield: Flowfield object to set the freestream.
    :param aoa: Angle of attack of the wing in radians. Defaults to 3 degrees.
    :param sweep: Sweep of the wing in radians. This just rotates the structure, and does not impact its properties
    otherwise.
    :param variable_disc_wake: If True, use a variable discretisation wake to extend the wake further downstream without
    an increase in panel count.
    :param custom_dt: Set a custom time step length. If None, uses the default time step length to enforce CFL=1.
    :return: CoupledAeroelastic Pazy wing model.
    """
    if variable_disc_wake:
        wake_delta = jnp.ones(m_star)
        wake_delta = wake_delta.at[2 * m :].set(jnp.logspace(0.0, 1.3, m_star - 2 * m))
    else:
        wake_delta = None

    match model:
        case "prepazy":
            data = PREPAZY_WITH_SKIN if skin else PREPAZY_NO_SKIN
        case "pazy":
            data = TECHNION_PAZY_WITH_SKIN if skin else TECHNION_PAZY_NO_SKIN
        case _:
            raise ValueError("Invalid model")

    return make_generic_pazy_wing(
        m=m,
        m_star=m_star,
        node_multiplier=node_multiplier,
        gravity=gravity,
        flowfield=flowfield,
        aoa=aoa,
        data=data,
        sweep=sweep,
        wake_delta=wake_delta,
        custom_dt=custom_dt,
    )


if __name__ == "__main__":
    r"""
    Run an example static Pazy case and print the tip deflection to console.
    """
    jax.config.update("jax_enable_x64", True)

    rho = 1.225
    aoa_ = jnp.deg2rad(7.0)
    u_inf_mag = 60.0

    wing_ = make_pazy_wing(
        gravity=True,
        node_multiplier=2,
        m=16,
        m_star=160,
        skin=True,
        aoa=aoa_,
        flowfield=Constant(
            rho=rho, u_inf=jnp.array((u_inf_mag, 0.0, 0.0)), relative_motion=True
        ),
    )
    static_sol = wing_.static_solve(
        prescribed_dofs=jnp.arange(6), horseshoe=False, load_steps=1, fsi_relaxation=0.5
    )
    static_sol.plot("./pazy_outputs/")

    z_tip = 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )
    print(f"Tip deflection at mid chord (m): {float(z_tip):.03f}")
    print(f"Relative tip deflection at mid chord (z/b) {float(z_tip) / 0.55:.03f}")
