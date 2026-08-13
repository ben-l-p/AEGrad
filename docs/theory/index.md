# Theory

Explanations of the mathematics and modelling choices behind Condor. Unlike the tutorials and how-tos, these pages are
discussion, not action — they exist to build intuition and to make design decisions traceable.

<div class="grid cards" markdown>

- **[Nonlinear beam](beam.md)**
  Geometrically-exact Cosserat beam on $\mathrm{SE} (3)$: strain measure, residual, tangent stiffness, and
  generalised-$\alpha$ time integration.

- **[UVLM](uvlm.md)**
  Unsteady vortex lattice method: bound and wake circulations, regularised Biot–Savart, non-penetration BC, wake
  convection, and force projection.

- **[Coupling](coupling.md)**
  Partitioned FSI with Aitken-accelerated Picard iteration; static and dynamic variants; convergence criteria.

- **[Linear](linear.md)**
  Linearisation about a nonlinear equilibrium: modal analysis, aerodynamic state-space, and the coupled aero-augmented
  eigenproblem used for flutter.

- **[Adjoint](adjoint.md)**
  Discrete adjoint for static and dynamic coupled problems; reuse of the primal tangent; checkpointing; design-variable
  ravel/unravel.

</div>
