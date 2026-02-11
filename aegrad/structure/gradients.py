from aegrad.structure.beam import BeamStructure
from aegrad.structure.data_structures import StaticStructure
from dataclasses import dataclass
from typing import Optional, Callable
from jax import Array


@dataclass
class BeamDesignVariables:
    r"""
    Data class to hold design variables for a beam structure. This includes the beam's cross-sectional properties and
    material properties, which can be optimized during the design process.
    """

    coords: Array
    k_cs: Array
    m_cs: Optional[Array]
    m_lumped: Optional[Array]


type ObjectiveFunction = Callable[[StaticStructure, BeamDesignVariables], Array]


class BeamGradients(BeamStructure):
    def static_adjoint(
        self, structure: StaticStructure, objective: ObjectiveFunction
    ) -> StaticStructure:
        r"""
        Computes the static adjoint of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param structure: StaticStructure containing the current state of the structure.
        :param objective: Objective function that takes the structure and design variables and returns a scalar loss.
        :return: Adjoint variables.
        """
