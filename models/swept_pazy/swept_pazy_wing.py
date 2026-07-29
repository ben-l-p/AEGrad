from typing import Literal

import jax
from jax import numpy as jnp
from jax import Array

from aegrad.aero.flowfields import FlowField, Constant
from aegrad.coupled import CoupledAeroelastic
from models.straight_pazy.pazy_wing import make_generic_pazy_wing


# constant from provided data
from models.swept_pazy.swept_pazy_properties import SWEEP_10_LE, SWEEP_10_TE


def make_swept_pazy_wing(
    m: int = 12,
    m_star: int = 120,
    sweep_angle: Literal[10] = 10,
    tip_mass: Literal["LE", "TE"] = "LE",
    node_multiplier: int = 1,
    gravity: bool | Array = False,
    flowfield: FlowField = Constant(
        u_inf=jnp.array((40.0, 0.0, 0.0)), rho=1.225, relative_motion=True
    ),
    aoa: float | Array = jnp.deg2rad(3.0),
) -> CoupledAeroelastic:
    match sweep_angle:
        case 10:
            match tip_mass:
                case "LE":
                    data = SWEEP_10_LE
                case "TE":
                    data = SWEEP_10_TE
                case _:
                    raise ValueError("Invalid tip_mass value")
        case _:
            raise ValueError("Invalid sweep_angle value")

    return make_generic_pazy_wing(
        m=m,
        m_star=m_star,
        node_multiplier=node_multiplier,
        gravity=gravity,
        flowfield=flowfield,
        aoa=aoa,
        data=data,
        sweep=None,
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    rho = 1.225
    aoa_ = jnp.deg2rad(7.0)
    u_inf_mag = 60.0

    wing_ = make_swept_pazy_wing(
        gravity=True,
        node_multiplier=2,
        m=16,
        m_star=160,
        sweep_angle=10,
        tip_mass="LE",
        aoa=aoa_,
        flowfield=Constant(
            rho=rho, u_inf=jnp.array((u_inf_mag, 0.0, 0.0)), relative_motion=True
        ),
    )
    static_sol = wing_.static_solve(
        prescribed_dofs=jnp.arange(6), horseshoe=False, load_steps=1, fsi_relaxation=0.5
    )
    static_sol.plot("./swept_pazy_outputs/")

    z_tip = 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )
    print(f"Tip deflection at mid chord (m): {float(z_tip):.03f}")
    print(f"Relative tip deflection at mid chord (z/b) {float(z_tip) / 0.55:.03f}")
