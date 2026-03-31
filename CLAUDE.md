# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AEGrad** is a JAX-based solver for nonlinear aeroelastic simulations with automatic differentiation. It couples:
- **UVLM** (Unsteady Vortex Lattice Method) for aerodynamics
- **Nonlinear beam theory** (SE(3)/SO(3) Lie group formulation) for structural mechanics
- **Adjoint-based gradients** for efficient sensitivity computation

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/structure/adjoint/test_two_node_beam_adjoint.py

# Run tests for a specific module
pytest tests/structure/
pytest tests/aero/
pytest tests/algebra/

# Lint / format
ruff check .
ruff format .
```

## Architecture

### Module Layout

```
aegrad/
├── algebra/        # Mathematical foundations: SO(3), SE(3), linear operators
├── aero/           # UVLM aerodynamics + influence coefficients + wake propagation
├── structure/      # Nonlinear beam solver (static + dynamic)
├── coupled/        # Aeroelastic coupling (static + dynamic, adjoint)
└── plotting/       # VTK output for ParaView visualization
models/             # Reusable problem factories (cantilever wing, flying spaghetti, etc.)
tests/              # Mirrors aegrad/ module structure
```

### JAX Pytree Pattern

All solver data classes use the `@_make_pytree` decorator (`aegrad/utils.py`) to register them as JAX pytrees. This enables `jax.grad`, `jax.vmap`, and `jax.vjp` to work across the entire state. Each class must implement `_dynamic_names()` (differentiable fields) and `_static_names()` (non-differentiable config). When adding new fields to any data class, they must be assigned to one of these two lists.

### Algebra Layer (`aegrad/algebra/`)

The structural solver uses Lie group formulations:
- **SO(3)** — rotation matrices, skew maps, exponential/log maps
- **SE(3)** — rigid body poses, tangent operators (used for beam kinematics)
- **base.py** — matrix exponential via Taylor series, finite differences for validation

### Structural Solver (`aegrad/structure/`)

- `beam.py` — `BeamStructure`: 6-DOF-per-node Cosserat beam, Newton iterations for static solve, Generalized-α time integration for dynamics
- `time_integration.py` — time-stepping logic; spectral radius parameter controls numerical damping
- `data_structures.py` — `StaticStructure`, `DynamicStructure`, `DynamicStructureSnapshot` hold state between solver calls
- `gradients/` — adjoint implementations for static and dynamic cases

### Aerodynamic Solver (`aegrad/aero/`)

- `uvlm.py` — core UVLM: assembles AIC matrices, solves for vortex strengths, propagates wake
- `aic.py` — aerodynamic influence coefficients (Biot-Savart kernel)
- `data_structures.py` — `GridDiscretization`, `DynamicAeroCase`, `AeroSnapshot`
- `gradients/` — AD-based sensitivities (JAX `vjp`/`jvp`)

### Coupled Solver (`aegrad/coupled/`)

- `gradients/coupled.py` — `CoupledAeroelastic`: top-level class with `.static_solve()`, `.dynamic_solve()`, and adjoint methods
- `data_structures.py` — `StaticAeroelastic`, `DynamicAeroelastic`

### Convergence Framework

`ConvergenceSettings` / `ConvergenceStatus` (`aegrad/data_structures.py`) are shared across structural and aeroelastic solvers. They track absolute/relative displacement and force residuals with configurable tolerances.

### Example Scripts

- `case.py` — static aeroelastic solve + adjoint gradient computation
- `spaghetti_adjoint.py` — dynamic adjoint with time integration
- `models/` — factory functions that return configured `CoupledAeroelastic` instances