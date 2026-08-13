# CONDOR - COupled Nonlinear Differentiable solver fOR aeroelastics

![Tests](https://github.com/ben-l-p/condor/actions/workflows/python_package.yml/badge.svg)
[![cov](https://ben-l-p.github.io/condor/badges/coverage.svg)](https://github.com/ben-l-p/condor/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://ben-l-p.github.io/condor/)
![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/ben-l-p/condor/main/pyproject.toml)

Condor is a differentiable nonlinear aeroelastic analysis framework which couples a nonlinear structural model with
unsteady vortex lattice method (UVLM) aerodynamics. This allows for a range of structural, aerodynamic and coupled
analyses, with gradients available using the adjoint method. The full codebase is implemented in JAX, which allows for
efficient automatic differentiation, GPU acceleration and multi-case parallelisation.

Full documentation, including tutorials, API reference and theory, is available at
[ben-l-p.github.io/condor](https://ben-l-p.github.io/condor/).

## Installation

Installation is available using pip - from the root of the repository, installation can be run with:

```bash
pip install .
```

It is reccomended to use ``uv`` as a virtual environment manager to manage dependencies. To install through ``uv``:

```bash
uv sync
```

An extensive test suite is included to verify the correctness of the code. This verified the numerics, and takes
approximately 30 minutes to run on an M2 MacBook Air. Tests can be run using pytest.

```bash
uv run pytest
```




