from __future__ import annotations

from _operator import mul
from dataclasses import dataclass
from functools import reduce
import os
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING, OrderedDict

import jax
from jax import Array, numpy as jnp

if TYPE_CHECKING:
    from aero.data_structures import DynamicAeroCase
from algebra.array_utils import ArrayList, ArrayListShape, vect_to_arrs
from plotting.aerogrid import plot_grid_to_vtk
from utils import _make_pytree
from data_structures import DesignVariables


@_make_pytree
class AeroStates:
    def __init__(self, gamma_b: ArrayList, gamma_w: ArrayList, gamma_b_dot: ArrayList, zeta_w: ArrayList) -> None:
        self.gamma_b: ArrayList = gamma_b
        self.gamma_w: ArrayList = gamma_w
        self.gamma_b_dot: ArrayList = gamma_b_dot
        self.zeta_w: ArrayList = zeta_w

    def shapes(self) -> OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]]:
        r"""
        Obtain the shapes of all arrays within the data structure.
        :return: Dictionary of name - shape pairs of all arrays within the data structure.
        """
        return OrderedDict(gamma_b=self.gamma_b.shape, gamma_w=self.gamma_w.shape, gamma_b_dot=self.gamma_b_dot.shape,
                           zeta_w=self.zeta_w.shape)

    @staticmethod
    def from_vector(vect: Array,
                    shapes: OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]]) -> AeroStates:
        return AeroStates(**vect_to_arrs(vect, shapes))

    def ravel(self) -> Array:
        r"""
        Ravel the data structure to a vector in a given order.
        :return: Data vector
        """

        return jnp.concatenate(
            [self.gamma_b.ravel(), self.gamma_w.ravel(), self.gamma_b_dot.ravel(), self.zeta_w.ravel()])

    @property
    def n_states(self) -> int:
        return self.gamma_b.size + self.gamma_w.size + self.gamma_b_dot.size + self.zeta_w.size

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ()

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "gamma_b", "gamma_w", "gamma_b_dot", "zeta_w"


@jax.tree_util.register_dataclass
@dataclass
class AeroStateGradients:
    d_f_steady_d_u: ArrayList
    d_f_unsteady_d_u: Optional[ArrayList]


@_make_pytree
class AeroDesignVariables(DesignVariables):
    def __init__(
            self,
            x0_aero: ArrayList,
            flowfield: dict[str, Array],
    ):
        super().__init__()
        self.x0_aero: ArrayList = x0_aero
        self.flowfield: dict[str, Array] = flowfield

        self.shapes: dict[
            str, Optional[tuple[int, ...] | ArrayListShape] | dict[str, tuple[int, ...] | ArrayListShape]] = (
            self.get_shapes()
        )
        self.mapping, self.n_x = self.make_index_mapping()

    def __iadd__(self, other: AeroDesignVariables) -> AeroDesignVariables:
        self.x0_aero = ArrayList([self.x0_aero[i] + other.x0_aero[i] for i in range(len(self.x0_aero))])
        for k in self.flowfield.keys():
            self.flowfield[k] += other.flowfield[k]
        return self

    def premult_adj(self, adj: Array) -> AeroDesignVariables:
        return AeroDesignVariables(
            x0_aero=ArrayList([jnp.einsum("ij,j...->i...", adj, self.x0_aero[i]) for i in range(len(self.x0_aero))]),
            flowfield={k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.flowfield.items()}
        )

    def get_vars(self) -> dict[str, Array | ArrayList | dict[str, Array]]:
        return {
            "x0_aero": self.x0_aero,
            "flowfield": self.flowfield
        }

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "x0_aero", "flowfield"


@_make_pytree
class AeroDesignGradients:
    def __init__(
            self,
            x0_aero: ArrayList,
            flowfield: dict[str, Array],
            f_shape: tuple[int, ...],
    ):
        self.x0_aero: ArrayList = x0_aero
        self.flowfield: dict[str, Array] = flowfield

        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = reduce(mul, f_shape, 1)

    def plot(
            self,
            case: DynamicAeroCase,
            rmat_nodal: Optional[ArrayList],
            directory: os.PathLike | str,
    ) -> Sequence[Path]:
        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        paths = []

        for i_surf in range(case.n_surf):
            bound_filename = Path(directory).joinpath(
                case.surf_b_names[i_surf] + "_gradient"
            )

            if rmat_nodal is not None:
                d_x0_aero: Array = jnp.einsum(
                    "ijk,lik->lij", rmat_nodal[i_surf], self.x0_aero[i_surf]
                )
            else:
                d_x0_aero = self.x0_aero[i_surf]

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
        return "x0_aero", "flowfield"
