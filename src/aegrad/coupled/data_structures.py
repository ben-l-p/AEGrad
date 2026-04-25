from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Optional, TYPE_CHECKING, Sequence, OrderedDict

import jax
from jax import Array, numpy as jnp

from aegrad.aero.data_structures import DynamicAeroCase, AeroSnapshot
from aegrad.aero.gradients.data_structures import AeroStates, AeroDesignVariables
from aegrad.algebra.array_utils import ArrayListShape, ArrayList
from aegrad.utils.data_structures import DesignVariables
from aegrad.utils.utils import make_pytree
from aegrad.structure.gradients.data_structures import StructuralDesignVariables
from aegrad.structure.data_structures import DynamicStructure, StructureMinimalStates

if TYPE_CHECKING:
    from aegrad.structure.data_structures import StaticStructure
    from aegrad.coupled.coupled import BaseCoupledAeroelastic
    from aegrad.structure.data_structures import (
        DynamicStructureSnapshot,
        StructureFullStates,
    )


class StaticAeroelastic:
    def __init__(self, structure: StaticStructure, aero: AeroSnapshot):
        self.structure: StaticStructure = structure
        self.aero: AeroSnapshot = aero

    def plot(
        self,
        directory: os.PathLike | str,
        n_interp: int = 0,
        plot_bound: bool = True,
        plot_wake: bool = True,
    ):
        self.structure.plot(directory, n_interp=n_interp)
        self.aero.plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)  # type: ignore

    def get_full_states(self):
        if self.structure.f_ext_aero is None:
            raise ValueError("f_ext_aero is None")

        return AeroelasticFullStates(
            structure=self.structure.get_full_states(),
            aero=self.aero.get_states(i_ts=0),
        )

    @overload
    def to_dynamic(self, t: None) -> DynamicAeroelasticSnapshot: ...

    @overload
    def to_dynamic(self, t: Array) -> DynamicAeroelastic: ...

    def to_dynamic(
        self, t: Optional[Array]
    ) -> DynamicAeroelasticSnapshot | DynamicAeroelastic:
        if t is None:
            struct = self.structure.to_dynamic(t=t)
            return DynamicAeroelasticSnapshot(structure=struct, aero=self.aero)
        else:
            struct = self.structure.to_dynamic(t=t)
            aero = self.aero.to_dynamic(i_ts=0, n_tstep=len(t) if t is not None else 1)
            return DynamicAeroelastic(structure=struct, aero=aero)


class DynamicAeroelasticSnapshot:
    def __init__(self, structure: DynamicStructureSnapshot, aero: AeroSnapshot):
        self.structure: DynamicStructureSnapshot = structure
        self.aero: AeroSnapshot = aero

    def to_static(self) -> StaticAeroelastic:
        return StaticAeroelastic(
            structure=self.structure.to_static(),
            aero=self.aero,
        )


class DynamicAeroelastic:
    def __init__(self, structure: DynamicStructure, aero: DynamicAeroCase):
        self.structure: DynamicStructure = structure
        self.aero: DynamicAeroCase = aero

    def __getitem__(self, i_ts: int) -> DynamicAeroelasticSnapshot:
        return DynamicAeroelasticSnapshot(
            structure=self.structure[i_ts], aero=self.aero[i_ts]
        )

    @classmethod
    def initialise(
        cls,
        initial_snapshot: DynamicAeroelasticSnapshot
        | DynamicAeroelastic
        | StaticAeroelastic,
        t: Array,
        use_f_ext_follower: bool,
        use_f_ext_dead: bool,
        aeroelastic_object: BaseCoupledAeroelastic,
    ) -> DynamicAeroelastic:
        if isinstance(initial_snapshot, DynamicAeroelasticSnapshot):
            init_struct: DynamicStructureSnapshot = initial_snapshot.structure
            init_aero: AeroSnapshot = initial_snapshot.aero
        elif isinstance(initial_snapshot, StaticAeroelastic):
            init_struct: DynamicStructureSnapshot = (
                initial_snapshot.structure.to_dynamic(t=None)
            )
            init_aero: AeroSnapshot = initial_snapshot.aero
        elif isinstance(initial_snapshot, DynamicAeroelastic):
            if initial_snapshot.structure.n_tstep != 1:
                raise ValueError("initial_snapshot.structure.n_tstep != 1")
            if initial_snapshot.aero.n_tstep != 1:
                raise ValueError("initial_snapshot.aero.n_tstep != 1")

            init_struct: DynamicStructureSnapshot = initial_snapshot.structure[0]
            init_aero: AeroSnapshot = initial_snapshot.aero[0]
        else:
            raise ValueError(
                "initial_snapshot must be DynamicAeroelastic or DynamicAeroCase"
            )

        struct_case = DynamicStructure.initialise(
            initial_snapshot=init_struct,
            t=t,
            use_f_ext_aero=True,
            use_f_ext_follower=use_f_ext_follower,
            use_f_ext_dead=use_f_ext_dead,
        )
        aero_case = DynamicAeroCase.initialise(
            initial_snapshot=init_aero, n_tstep=len(t)
        )

        # compute aerodynamic forcing at timestep 0
        f_aero_init = aero_case.project_forcing_to_beam(
            i_ts=0,
            rmat=struct_case.hg[0, :, :3, :3],
            x0_aero=aeroelastic_object.aero.x0_b,
            include_unsteady=False,
        )
        f_aero_local = aeroelastic_object.structure.make_f_dead_ext(
            f_ext=f_aero_init, rmat=struct_case.hg[0, :, :3, :3]
        )

        if struct_case.f_ext_aero is None:
            raise ValueError("f_ext_aero cannot be None")

        struct_case.f_ext_aero = struct_case.f_ext_aero.at[0, ...].set(f_aero_local)

        return DynamicAeroelastic(structure=struct_case, aero=aero_case)

    def get_full_states(self, i_ts: int | Array) -> AeroelasticFullStates:
        r"""
        Get the full aeroelastic states for the system at a given timestep.
        :param i_ts: Time step index .
        :return: Full aeroelastic states.
        """
        return AeroelasticFullStates(
            structure=self.structure.get_full_states(i_ts=i_ts),
            aero=self.aero.get_states(i_ts=i_ts),
        )

    def get_minimal_states(self, i_ts: int | Array) -> AeroelasticMinimalStates:
        r"""
        Get the minimal aeroelastic states for the system at a given timestep.
        :param i_ts: Time step index.
        :return: Minimal aeroelastic states.
        """

        return AeroelasticMinimalStates(
            structure=self.structure.get_minimal_states(i_ts=i_ts),
            aero=self.aero.get_states(i_ts=i_ts),
        )

    def plot(
        self,
        directory: os.PathLike | str,
        index: Optional[int | Sequence[int] | Array | slice] = None,
        n_interp: int = 0,
        plot_bound: bool = True,
        plot_wake: bool = True,
    ) -> tuple[Path, Sequence[Path]]:
        r"""
        Plots the aeroelastic dynamic system.
        :param directory: Directory to save the plots to.
        :param index: Index of timesteps to plot.
        :param n_interp: Number of interpolation points for plotting the structure.
        :param plot_bound: Whether to plot the bound aerodynamic panels.
        :param plot_wake: Whether to plot the wake aerodynamic panels.
        :return: Paths of the structural and aerodynamic PVD files.
        """
        struct_pvd: Path = self.structure.plot(
            directory=directory, n_interp=n_interp, index=index
        )
        aero_pvd: Sequence[Path] = self.aero.plot(
            directory=directory,  # type: ignore
            plot_bound=plot_bound,
            plot_wake=plot_wake,
            index=index,
        )
        return struct_pvd, aero_pvd


@jax.tree_util.register_dataclass
@dataclass
class AeroelasticFullStates:
    aero: AeroStates
    structure: StructureFullStates


@make_pytree
class AeroelasticMinimalStates:
    def __init__(self, structure: StructureMinimalStates, aero: AeroStates):
        self.structure: StructureMinimalStates = structure
        self.aero: AeroStates = aero

    @staticmethod
    def from_vector(
        vect: Array,
        n_dof: int,
        aero_shapes: OrderedDict[str, Optional[tuple[int, ...] | ArrayListShape]],
    ) -> AeroelasticMinimalStates:
        struct = StructureMinimalStates.from_mat(vect[: 5 * n_dof].reshape(5, n_dof))
        aero = AeroStates.from_vector(vect[5 * n_dof :], aero_shapes)
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


@make_pytree
class AeroelasticDesignVariables(DesignVariables):
    def __init__(
        self,
        structure_dv: StructuralDesignVariables,
        aero_dv: AeroDesignVariables,
    ):
        super().__init__()
        self.structure: StructuralDesignVariables = structure_dv
        self.aero: AeroDesignVariables = aero_dv

        self.shapes: OrderedDict[
            str,
            Optional[
                tuple[int, ...]
                | ArrayListShape
                | OrderedDict[str, tuple[int, ...] | ArrayListShape]
            ],
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def get_vars(self) -> dict[str, Optional[Array]]:
        return {
            **(self.structure.get_vars() if self.structure is not None else {}),
            **(self.aero.get_vars() if self.aero is not None else {}),
        }

    def split_adjoint(
        self, d_f_d_x: dict[str, Optional[Array | ArrayList]], f_shape: tuple[int, ...]
    ) -> AeroelasticDesignVariables:
        struct_dv = StructuralDesignVariables(
            **{k: v for k, v in d_f_d_x.items() if k in self.structure.get_vars()},
            f_shape=f_shape,
        )
        aero_dv = AeroDesignVariables(
            **{k: v for k, v in d_f_d_x.items() if k in self.aero.get_vars()},
            f_shape=f_shape,
        )
        return AeroelasticDesignVariables(structure_dv=struct_dv, aero_dv=aero_dv)

    def premultiply_adj(self, adj: Array) -> AeroelasticDesignVariables:
        return AeroelasticDesignVariables(
            structure_dv=self.structure.premultiply_adj(adj),
            aero_dv=self.aero.premultiply_adj(adj),
        )

    def __iadd__(self, other: AeroelasticDesignVariables) -> AeroelasticDesignVariables:
        self.structure += other.structure
        self.aero += other.aero
        return self

    def plot(
        self,
        case: StaticAeroelastic | DynamicAeroelastic,
        i_ts: Optional[int],
        directory: os.PathLike | str,
    ) -> Sequence[Path]:
        paths = []

        if isinstance(case, StaticAeroelastic):
            struct_snapshot: StaticStructure | DynamicStructureSnapshot = case.structure
            aero_snapshot: AeroSnapshot = case.aero
        elif isinstance(case, DynamicAeroelastic):
            if i_ts is None:
                raise ValueError("Time step index must be specified")
            struct_snapshot = case.structure[i_ts]
            aero_snapshot: AeroSnapshot = case.aero[i_ts]
        else:
            raise ValueError("Case must be StaticAeroelastic or DynamicAeroelastic")

        if self.structure is not None:
            paths.append(
                self.structure.plot(
                    case=struct_snapshot, directory=directory, n_interp=0
                )
            )
        if self.aero is not None:
            rmat_nodal = ArrayList(
                [struct_snapshot.hg[mp, :3, :3] for mp in case.aero.dof_mapping]
            )
            paths.extend(
                self.aero.plot(
                    aero_snapshot, directory=directory, rmat_nodal=rmat_nodal
                )
            )
        return paths

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "structure", "aero"
