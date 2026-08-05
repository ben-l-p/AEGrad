from jax import numpy as jnp

from aegrad.aero.flowfields import Constant
from models.pazy.straight.pazy_wing import make_pazy_wing


class TestPazyModal:
    @staticmethod
    def test_modes():
        r"""
        Check the first 5 modes of the Pazy model, and that they are close to literature. Using the model from
        https://github.com/UM-A2SRL/AePW3-LDWG/tree/main/00_Models/01_Pazy_Technion/02_Models_Beam and comparing with
        modes presented in "Collaborative Pazy Wing Analyses for the Third Aeroelastic Prediction Workshop".
        """

        wing = make_pazy_wing(
            flowfield=Constant(
                u_inf=jnp.array([0.0, 0.0, 0.0]),
                rho=1.225,
                relative_motion=True,
            ),
            aoa=jnp.deg2rad(0.0),
            m=6,
            m_star=0,
            node_multiplier=2,
            skin=True,
            variable_disc_wake=False,
            sweep=0.0,
        )

        reference = wing.reference_configuration(prescribed_dofs=tuple(range(6)))

        freqs, *_ = wing.structure.modal(
            case=reference.structure,
            n_modes=5,
        )

        literature_freqs = jnp.array((4.19, 28.47, 40.74, 82.39, 107.78))  # hz

        # torsion mode has a larger discrepancy but this is expected - we apply per-mode tolerancing from 1-4%
        tolerances = jnp.array((1e-2, 2e-2, 4e-2, 2e-2, 1e-2))

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )
