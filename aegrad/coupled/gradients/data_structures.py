from __future__ import annotations

from _operator import mul
from dataclasses import dataclass
from functools import reduce
import os
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING, OrderedDict

import jax
from jax import Array
from jax import numpy as jnp

from aero.gradients.data_structures import (
    AeroDesignVariables,
    AeroDesignGradients,
    AeroStateGradients, AeroStates,
)
from structure.data_structures import StructureMinimalStates
from structure.gradients.data_structures import StructuralStateGradients
from algebra.array_utils import ArrayList, ArrayListShape

if TYPE_CHECKING:
    from coupled.data_structures import StaticAeroelastic
    from structure import StructureFullStates, StructuralDesignVariables
from structure.gradients.data_structures import StructureDesignGradients
from utils import _make_pytree
from data_structures import DesignVariables


@jax.tree_util.register_dataclass
@dataclass
class AeroelasticFullStates:
    aero: AeroStates
    structure: StructureFullStates


@_make_pytree
class AeroelasticMinimalStates:
    def __init__(self, structure: StructureMinimalStates, aero: AeroStates):
        self.structure: StructureMinimalStates = structure
        self.aero: AeroStates = aero

    @staticmethod
    def from_vector(vect: Array, n_dof: int,
                    aero_shapes: OrderedDict[
                        str, Optional[tuple[int, ...] | ArrayListShape]]) -> AeroelasticMinimalStates:
        struct = StructureMinimalStates.from_mat(vect[:5 * n_dof].reshape(5, n_dof))
        aero = AeroStates.from_vector(vect[5 * n_dof:], aero_shapes)
        return AeroelasticMinimalStates(structure=struct, aero=aero)

    def ravel(self) -> Array:
        return jnp.concatenate([self.structure.ravel(), self.aero.ravel()])

    @property
    def n_states(self) -> int:
        return self.structure.n_states + self.aero.n_states

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ()

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "structure", "aero"


@jax.tree_util.register_dataclass
@dataclass
class AeroelasticStateGradients:
    aero: AeroStateGradients
    structure: StructuralStateGradients


@_make_pytree
class AeroelasticDesignVariables(DesignVariables):
    def __init__(
            self,
            structure_dv: StructuralDesignVariables,
            aero_dv: AeroDesignVariables,
    ):
        super().__init__()
        self.structure: StructuralDesignVariables = structure_dv
        self.aero: AeroDesignVariables = aero_dv

        self.shapes: OrderedDict[str, Optional[
            tuple[int, ...] | ArrayListShape | OrderedDict[str, tuple[int, ...] | ArrayListShape]]] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def get_vars(self) -> dict[str, Optional[Array]]:
        return {
            **(self.structure.get_vars() if self.structure is not None else {}),
            **(self.aero.get_vars() if self.aero is not None else {}),
        }

    def split_adjoint(
            self, d_f_d_x: dict[str, Optional[Array | ArrayList]], f_shape: tuple[int, ...]
    ) -> AeroelasticDesignGradients:
        struct_dv = StructureDesignGradients(
            **{k: v for k, v in d_f_d_x.items() if k in self.structure.get_vars()},
            f_shape=f_shape,
        )
        aero_dv = AeroDesignGradients(
            **{k: v for k, v in d_f_d_x.items() if k in self.aero.get_vars()},
            f_shape=f_shape,
        )
        return AeroelasticDesignGradients(
            structure_dv=struct_dv, aero_dv=aero_dv, f_shape=f_shape
        )

    def premult_adj(self, adj: Array) -> AeroelasticDesignVariables:
        return AeroelasticDesignVariables(
            structure_dv=self.structure.premult_adj(adj),
            aero_dv=self.aero.premult_adj(adj),
        )

    def __iadd__(self, other: AeroelasticDesignVariables) -> AeroelasticDesignVariables:
        self.structure += other.structure
        self.aero += other.aero
        return self

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "structure", "aero"


@_make_pytree
class AeroelasticDesignGradients:
    def __init__(
            self,
            structure_dv: StructureDesignGradients,
            aero_dv: AeroDesignGradients,
            f_shape: tuple[int, ...],
    ):
        self.structure: StructureDesignGradients = structure_dv
        self.aero: AeroDesignGradients = aero_dv
        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = reduce(mul, f_shape, 1)

    def plot(
            self, case: StaticAeroelastic, directory: os.PathLike | str
    ) -> Sequence[Path]:
        paths = []
        if self.structure is not None:
            paths.append(
                self.structure.plot(case.structure, directory=directory, n_interp=0)
            )
        if self.aero is not None:
            rmat_nodal = ArrayList(
                [case.structure.hg[mp, :3, :3] for mp in case.aero.dof_mapping]
            )
            paths.extend(
                self.aero.plot(case.aero, directory=directory, rmat_nodal=rmat_nodal)
            )
        return paths

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "f_shape", "f_size"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "structure", "aero"
