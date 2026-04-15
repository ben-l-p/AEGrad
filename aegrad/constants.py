# minimum value of ||h_omega||^2 before reverting to small angle approximation for SE(3)/SO(3)
# functions with singularities (exp, log, tangent and inverse tangent)
SMALL_ANG_THRESH = 1e-10

# summation order for computing operators from infinite series for comparing against full solution in tests
BASE_SUMMATION_ORDER = 15

# length of horseshoe wake, in meters
HORSESHOE_LENGTH = 1e3

# small number to avoid division by zero in Biot-Savart kernel with epsilon regularisation
EPSILON = 1e-7

# cutoff radius for Biot-Savart kernel, in meters
R_CUTOFF = 1e-4

# batch size for limiting memory usage for AIC computation
AIC_BATCH_SIZE = 2
