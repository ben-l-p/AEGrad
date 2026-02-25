from __future__ import annotations

from os import PathLike
from dataclasses import dataclass

from aegrad.aero import StaticAero, DynamicAero
from aegrad.structure import StaticStructure, DynamicStructure


class StaticAeroelastic:
    def __init__(self, structure: StaticStructure, aero: StaticAero):
        self.structure: StaticStructure = structure
        self.aero: StaticAero = aero

    def plot(
        self,
        directory: PathLike | str,
        n_interp: int = 0,
        plot_bound: bool = True,
        plot_wake: bool = True,
    ):
        self.structure.plot(directory, n_interp=n_interp)
        self.aero.plot(directory, plot_bound=plot_bound, plot_wake=plot_wake)


@dataclass
class DynamicAeroelastic:
    structure: DynamicStructure
    aero: DynamicAero
