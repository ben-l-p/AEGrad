from flapjax.aero.data_structures import (
    AeroCase,
    GridDiscretisation,
)
from flapjax.aero.flowfields import ConstantFlowField, OneMinusCosineFlowField
from flapjax.aero.gradients.data_structures import (
    AeroGradsToCompute,
    AeroJacobianApproximations,
    AeroStates,
)
from flapjax.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroLinearResult,
    AeroOutputUnflattened,
    AeroStateUnflattened,
)
from flapjax.aero.linear.linear_uvlm import LinearUVLM
from flapjax.aero.utils import add_control_surface, make_rectangular_grid
from flapjax.aero.uvlm import UVLM
