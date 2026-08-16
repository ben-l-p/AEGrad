from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from jax import Array
from jax import numpy as jnp

from flapjax.aero.gradients.data_structures import (
    AeroGradsToCompute,
    AeroJacobianApproximations,
)
from flapjax.structure.gradients.data_structures import (
    BeamJacobianApproximations,
    StructuralGradsToCompute,
)
from flapjax.utils.print_utils import jax_print
from flapjax.utils.utils import make_pytree


@dataclass(frozen=True)
class AeroelasticGradsToCompute:
    structure: StructuralGradsToCompute = field(
        default_factory=StructuralGradsToCompute
    )
    aero: AeroGradsToCompute = field(default_factory=AeroGradsToCompute)


@dataclass
class AeroelasticJacobianApproximations:
    structure: BeamJacobianApproximations = field(
        default_factory=BeamJacobianApproximations
    )
    aero: AeroJacobianApproximations = field(default_factory=AeroJacobianApproximations)


@make_pytree
class TrimVariables:
    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        cs_ang: dict[str, Array],
        thrust: dict[str, Array],
        trim_angles: dict[str, Array],
    ):
        self.cs_ang: dict[str, Array] = cs_ang
        self.thrust: dict[str, Array] = thrust
        self.trim_angles: dict[str, Array] = trim_angles

    def print(self, i_iter: int, f_clamp: Array | None) -> None:
        val_w = 5  # fixed character width per rendered float value
        inner_width = 81  # total width of the content inside the borders

        # Build rows as (label, format_string, rendered_width, kwargs)
        rows: list[tuple[str, str, int, dict[str, Array]]] = []

        def _entries(keys_):
            return "  ".join(f"[{k}]: {{{k}:>{val_w}.2f}}" for k in keys_)

        def _entries_rendered_len(keys_):
            return sum(len(k) + 4 + val_w for k in keys_) + 2 * (len(keys_) - 1)

        iter_label = "Iteration               "
        padding = inner_width - len(iter_label) - val_w - 4
        jax_print(
            f"| {iter_label} | {{i_iter:>{val_w}}}" + " " * padding + "|",
            i_iter=i_iter,
            verbose_level="normal",
        )

        # jax.debug.print's numeric format for printing required 0-D values
        def _scalar(x: Array) -> Array:
            return jnp.ravel(x)[0]

        if self.cs_ang:
            label = "Control surface angles (deg)"
            rows.append(
                (
                    label,
                    _entries(self.cs_ang),
                    _entries_rendered_len(self.cs_ang),
                    {k: jnp.rad2deg(_scalar(v)) for k, v in self.cs_ang.items()},
                )
            )

        if self.thrust:
            label = "Thrust force (N)        "
            rows.append(
                (
                    label,
                    _entries(self.thrust),
                    _entries_rendered_len(self.thrust),
                    {k: _scalar(v) for k, v in self.thrust.items()},
                )
            )

        if self.trim_angles:
            label = "Body Euler angles (deg) "
            rows.append(
                (
                    label,
                    _entries(self.trim_angles),
                    _entries_rendered_len(self.trim_angles),
                    {k: jnp.rad2deg(_scalar(v)) for k, v in self.trim_angles.items()},
                )
            )

        if f_clamp is not None:
            label = "Residual clamp force (N)"
            keys = [f"f{i}" for i in range(f_clamp.shape[0])]
            entries = "  ".join(f"[{k}]: {{{k}:>{val_w}.2f}}" for k in keys)
            rendered_len = _entries_rendered_len(keys)
            kwargs = {k: _scalar(f_clamp[i]) for i, k in enumerate(keys)}
            rows.append((label, entries, rendered_len, kwargs))

        for label, fmt, rendered_len, kwargs in rows:
            content = f"| {label} | {fmt}"
            padding = inner_width - len(label) - rendered_len - 4
            content += " " * padding + "|"
            jax_print(content, **kwargs, verbose_level="normal")

