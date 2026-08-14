from typing import Literal

from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import Constant, FlowField
from flapjax.coupled import CoupledAeroelastic
from flapjax.models.pazy.base import make_generic_pazy_wing

# constant from provided data
from flapjax.models.pazy.swept.data.properties import (
    SWEEP_10_LE,
    SWEEP_10_LE_CORRECTED,
    SWEEP_10_TE,
    SWEEP_10_TE_CORRECTED,
)

DEFAULT_FLOWFIELD: FlowField = Constant(
    u_inf=jnp.array((40.0, 0.0, 0.0)), rho=1.225, relative_motion=True
)
DEFAULT_AOA: Array = jnp.deg2rad(3.0)


def make_swept_pazy_wing(
    m: int = 12,
    m_star: int = 120,
    sweep_angle: Literal[10] = 10,
    tip_mass: Literal["LE", "TE", "LE_CORRECTED", "TE_CORRECTED"] = "LE_CORRECTED",
    node_multiplier: int = 1,
    gravity: bool | Array = False,
    flowfield: FlowField = DEFAULT_FLOWFIELD,
    aoa: float | Array = DEFAULT_AOA,
    variable_disc_wake: bool = False,
) -> CoupledAeroelastic:
    match sweep_angle:
        case 10:
            match tip_mass:
                case "LE":
                    data = SWEEP_10_LE
                case "TE":
                    data = SWEEP_10_TE
                case "LE_CORRECTED":
                    data = SWEEP_10_LE_CORRECTED
                case "TE_CORRECTED":
                    data = SWEEP_10_TE_CORRECTED
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
        sweep=None,  # if sweep were to be added here, it would be added on top of the sweep from data
        variable_disc_wake=variable_disc_wake,
    )
