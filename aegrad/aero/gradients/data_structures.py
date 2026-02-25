from __future__ import annotations

from _operator import mul
from dataclasses import dataclass
from functools import reduce
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

import jax
from jax import Array, numpy as jnp

from aero import StaticAero
from algebra.array_utils import ArrayList, ArrayListShape
from plotting.aerogrid import plot_grid_to_vtk
from utils import _make_pytree
from data_structures import DesignVariables


@jax.tree_util.register_dataclass
@dataclass
class AeroStates:
    f_steady: ArrayList
    f_unsteady: Optional[ArrayList]
    gamma_b: ArrayList
    gamma_w: ArrayList


@_make_pytree
class AeroDesignVariables(DesignVariables):
    def __init__(
        self,
        x0_aero: Optional[ArrayList | Sequence[Array] | Array],
        u_inf: Optional[Array],
        rho: Optional[Array],
    ):
        super().__init__()
        self.x0_aero: Optional[ArrayList] = (
            ArrayList(x0_aero) if x0_aero is not None else None
        )
        self.u_inf: Optional[Array] = u_inf
        self.rho: Optional[Array] = rho

        self.shapes: dict[str, Optional[tuple[int, ...] | ArrayListShape]] = (
            self.get_shapes()
        )
        self.mapping, self.n_x = self.make_index_mapping()

    def get_vars(self) -> dict[str, Optional[Array]]:
        return {
            "x0_aero": self.x0_aero,
            "u_inf": self.u_inf,
            "rho": self.rho,
        }

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "x0_aero", "u_inf", "rho"


@_make_pytree
class AeroDesignGradients:
    def __init__(
        self,
        x0_aero: Optional[ArrayList | Sequence[Array] | Array],
        u_inf: Optional[Array],
        rho: Optional[Array],
        f_shape: tuple[int, ...],
    ):
        self.x0_aero: Optional[ArrayList] = (
            ArrayList(x0_aero) if x0_aero is not None else None
        )
        self.u_inf: Optional[Array] = u_inf
        self.rho: Optional[Array] = rho

        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = reduce(mul, f_shape, 1)

    def plot(
        self,
        case: StaticAero,
        rmat_nodal: Optional[ArrayList],
        directory: PathLike | str,
    ) -> Sequence[Path]:
        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []

        for i_surf in range(case.n_surf):
            bound_filename = Path(directory).joinpath(
                case.surf_b_names[i_surf] + "_gradient"
            )

            if self.x0_aero is not None:
                if rmat_nodal is not None:
                    d_x0_aero = jnp.einsum(
                        "ijk,lik->lij", rmat_nodal[i_surf], self.x0_aero[i_surf]
                    )
                else:
                    d_x0_aero = self.x0_aero
            else:
                d_x0_aero = None

            paths.append(
                plot_grid_to_vtk(
                    case.zeta_b[i_surf],
                    bound_filename,
                    None,
                    node_vector_data={
                        "x0_aero": d_x0_aero,
                    },
                    cell_scalar_data={},
                )
            )

        return paths

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "f_shape", "f_size"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "x0_aero", "u_inf", "rho"
