from condor.structure.data_structures import (
    DynamicStructure,
    DynamicStructureSnapshot,
    OptionalJacobians,
    StaticStructure,
)
from condor.structure.gradients.beam import BeamStructure
from condor.structure.gradients.data_structures import (
    BeamJacobianApproximations,
    StructuralDesignVariables,
    StructuralGradsToCompute,
    StructureFullStates,
)
from condor.structure.linear.data_structures import (
    BeamInputUnflattened,
    BeamLinearResult,
    BeamOutputUnflattened,
    BeamStateUnflattened,
)
from condor.structure.linear.linear_beam import LinearBeam
