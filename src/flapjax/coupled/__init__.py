from flapjax.coupled.data_structures import (
    AeroelasticCase,
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.coupled.gradients.coupled import CoupledAeroelastic
from flapjax.coupled.gradients.data_structures import (
    AeroelasticGradsToCompute,
    TrimVariables,
)
from flapjax.coupled.linear.data_structures import (
    AeroelasticInputUnflattened,
    AeroelasticLinearResult,
    AeroelasticOutputUnflattened,
    AeroelasticStateUnflattened,
)
from flapjax.coupled.linear.linear_coupled import LinearCoupled
