from __future__ import annotations
from typing import Sequence

from dataclasses import dataclass, field

from jax import Array
from jax import numpy as jnp

from aegrad.aero.gradients.data_structures import AeroGradsToCompute
from aegrad.structure.gradients.data_structures import StructuralGradsToCompute
from aegrad.utils.utils import make_pytree
from aegrad.utils.print_utils import jax_print, VerbosityLevel


@dataclass(frozen=True)
class AeroelasticGradsToCompute:
    structure: StructuralGradsToCompute = field(
        default_factory=StructuralGradsToCompute
    )
    aero: AeroGradsToCompute = field(default_factory=AeroGradsToCompute)


@make_pytree
class TrimVariables:
    def __init__(
        self,
        cs_ang: dict[str, Array],
        thrust: dict[str, Array],
        trim_angles: dict[str, Array],
    ):
        self.cs_ang: dict[str, Array] = cs_ang
        self.thrust: dict[str, Array] = thrust
        self.trim_angles: dict[str, Array] = trim_angles

    def print(self) -> None:
        jax_print(
            "Control surface angles (deg): {cs_ang}",
            cs_ang={k: jnp.rad2deg(v) for k, v in self.cs_ang.items()},
            verbose_level=VerbosityLevel.NORMAL,
        )
        jax_print(
            "Thrust: {thrust}", thrust=self.thrust, verbose_level=VerbosityLevel.NORMAL
        )
        jax_print(
            "Trim angles (deg): {trim_angles}",
            trim_angles={k: jnp.rad2deg(v) for k, v in self.trim_angles.items()},
            verbose_level=VerbosityLevel.NORMAL,
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ()

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "cs_ang", "thrust", "trim_angles"
