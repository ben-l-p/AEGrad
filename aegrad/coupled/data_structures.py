from __future__ import annotations

import os
from pathlib import Path
from typing import overload, Optional, TYPE_CHECKING, Sequence

from jax import Array

from aero.data_structures import DynamicAeroCase, AeroSnapshot

if TYPE_CHECKING:
    from structure import DynamicStructureSnapshot
    from structure.data_structures import StaticStructure
    from coupled.coupled import BaseCoupledAeroelastic
from structure.data_structures import DynamicStructure

from coupled.gradients.data_structures import AeroelasticStates


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
        return AeroelasticStates(structure=self.structure.get_full_states(), aero=self.aero.get_full_states())

    @overload
    def to_dynamic(self, t: None) -> DynamicAeroelasticSnapshot:
        ...

    @overload
    def to_dynamic(self, t: Array) -> DynamicAeroelastic:
        ...

    def to_dynamic(self, t: Optional[Array]) -> DynamicAeroelasticSnapshot | DynamicAeroelastic:
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
        return DynamicAeroelasticSnapshot(structure=self.structure[i_ts], aero=self.aero[i_ts])

    @classmethod
    def initialise(cls,
                   initial_snapshot: DynamicAeroelasticSnapshot | DynamicAeroelastic | StaticAeroelastic,
                   t: Array,
                   use_f_ext_follower: bool,
                   use_f_ext_dead: bool,
                   aeroelastic_object: BaseCoupledAeroelastic) -> DynamicAeroelastic:
        if isinstance(initial_snapshot, DynamicAeroelasticSnapshot):
            init_struct: DynamicStructureSnapshot = initial_snapshot.structure
            init_aero: AeroSnapshot = initial_snapshot.aero
        elif isinstance(initial_snapshot, StaticAeroelastic):
            init_struct: DynamicStructureSnapshot = initial_snapshot.structure.to_dynamic(t=None)
            init_aero: AeroSnapshot = initial_snapshot.aero
        elif isinstance(initial_snapshot, DynamicAeroelastic):
            if initial_snapshot.structure.n_tstep != 1: raise ValueError("initial_snapshot.structure.n_tstep != 1")
            if initial_snapshot.aero.n_tstep != 1: raise ValueError("initial_snapshot.aero.n_tstep != 1")

            init_struct: DynamicStructureSnapshot = initial_snapshot.structure[0]
            init_aero: AeroSnapshot = initial_snapshot.aero[0]
        else:
            raise ValueError("initial_snapshot must be DynamicAeroelastic or DynamicAeroCase")

        struct_case = DynamicStructure.initialise(initial_snapshot=init_struct, t=t, use_f_ext_aero=True,
                                                  use_f_ext_follower=use_f_ext_follower, use_f_ext_dead=use_f_ext_dead)
        aero_case = DynamicAeroCase.initialise(initial_snapshot=init_aero, n_tstep=len(t))

        # compute aerodynamic forcing at timestep 0
        f_aero_init = aero_case.project_forcing_to_beam(i_ts=0, rmat=struct_case.hg[0, :, :3, :3],
                                                        x0_aero=aeroelastic_object.aero.x0_b,
                                                        include_unsteady=False)
        f_aero_local = aeroelastic_object.structure.make_f_dead_ext(f_ext=f_aero_init,
                                                                    rmat=struct_case.hg[0, :, :3, :3])

        if struct_case.f_ext_aero is None: raise ValueError("f_ext_aero cannot be None")

        struct_case.f_ext_aero = struct_case.f_ext_aero.at[0, ...].set(f_aero_local)

        return DynamicAeroelastic(structure=struct_case, aero=aero_case)

    def plot(self, directory: os.PathLike | str, index: Optional[int | Sequence[int] | Array | slice] = None,
             n_interp: int = 0,
             plot_bound: bool = True, plot_wake: bool = True) -> tuple[Path, Sequence[Path]]:
        r"""
        Plots the aeroelastic dynamic system.
        :param directory: Directory to save the plots to.
        :param index: Index of timesteps to plot.
        :param n_interp: Number of interpolation points for plotting the structure.
        :param plot_bound: Whether to plot the bound aerodynamic panels.
        :param plot_wake: Whether to plot the wake aerodynamic panels.
        :return: Paths of the structural and aerodynamic PVD files.
        """
        struct_pvd: Path = self.structure.plot(directory=directory, n_interp=n_interp, index=index)
        aero_pvd: Sequence[Path] = self.aero.plot(directory=directory, plot_bound=plot_bound, plot_wake=plot_wake,
                                                  index=index)
        return struct_pvd, aero_pvd
