# AEGrad - Differentiable Nonlinear Aeroelastic Analysis

AEGrad is a differentiable nonlinear aeroelastic analysis framework which couples a nonlinear structural model with
unsteady vortex lattice method (UVLM) aerodynamics. This allows for a range of structural, aerodynamic and coupled
analyses, with gradients available using the adjoint method. The full codebase is implemented in JAX, which allows for
efficient automatic differentiation and GPU acceleration.

## Installation

Installation is available using pip. It is recommended to use UV as a virtual environment for installation.

```bash
pip install .
```

An extensive test suite is included to verify the correctness of the code. This verified the numerics, and takes
approximately 15 minutes to run on an M2 MacBook air. Tests can be run using pytest.

```bash
uv run pytest
```




