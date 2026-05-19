from dataclasses import dataclass, field

from aegrad.aero.gradients.data_structures import AeroGradsToCompute
from aegrad.structure.gradients.data_structures import StructuralGradsToCompute


@dataclass(frozen=True)
class AeroelasticGradsToCompute:
    structure: StructuralGradsToCompute = field(default_factory=StructuralGradsToCompute)
    aero: AeroGradsToCompute = field(default_factory=AeroGradsToCompute)
