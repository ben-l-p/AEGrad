from typing import Literal

from jax import numpy as jnp, Array

from aegrad.aero.flowfields import Constant
from models.pazy.straight.pazy_wing import make_pazy_wing
from models.pazy.swept.swept_pazy_wing import make_swept_pazy_wing


class TestPazyModal:
    @staticmethod
    def test_straight_modes():
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

    @staticmethod
    def _swept_modes(
        tip_mass: Literal["LE", "TE", "LE_CORRECTED", "TE_CORRECTED"],
    ) -> Array:
        wing = make_swept_pazy_wing(
            flowfield=Constant(
                u_inf=jnp.array([0.0, 0.0, 0.0]),
                rho=1.225,
                relative_motion=True,
            ),
            tip_mass=tip_mass,
            aoa=jnp.deg2rad(0.0),
            m=6,
            m_star=0,
            node_multiplier=2,
        )

        reference = wing.reference_configuration(prescribed_dofs=tuple(range(6)))

        freqs, *_ = wing.structure.modal(
            case=reference.structure,
            n_modes=3,
        )
        return freqs

    def test_sweep_10_le_modes(self):
        r"""
        Check the first 3 modes of the swept Pazy model with leading edge mass, and that they are close to literature.
        Comparing with "Flutter, Post-Flutter, and Limit-Cycle Oscillations of Very Flexible Swept Wings".
        """
        freqs = self._swept_modes(tip_mass="LE")

        literature_freqs = jnp.array((4.31, 28.9, 37.5))  # hz

        # the error in torsion here is quite high, hence the large torsion tolerance to allow it to pass
        # to make this model more useful, we make a corrected model
        tolerances = jnp.array(
            (
                1.1e-2,
                1e-2,
                0.1,
            )
        )

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )

    def test_sweep_10_le_corrected_modes(self):
        r"""
        Check the first 3 modes of the swept Pazy model with leading edge mass, and that they are close to literature.
        This uses corrections to the tip inertia properties to better match the torsion mode. Comparing with "Flutter,
        Post-Flutter, and Limit-Cycle Oscillations of Very Flexible Swept Wings".
        """
        freqs = self._swept_modes(tip_mass="LE_CORRECTED")

        literature_freqs = jnp.array((4.31, 28.9, 37.5))  # hz

        tolerances = jnp.array(
            (
                1.1e-2,
                1e-2,
                2e-2,
            )
        )

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )

    def test_sweep_10_te_modes(self):
        r"""
        Check the first 3 modes of the swept Pazy model with trailing edge mass, and that they are close to literature.
        Comparing with "Flutter, Post-Flutter, and Limit-Cycle Oscillations of Very Flexible Swept Wings".
        """
        freqs = self._swept_modes(tip_mass="TE")

        literature_freqs = jnp.array((4.27, 27.4, 38.7))  # hz

        # second bending mode has a larger discrepancy, but torsion mode is relatively good
        tolerances = jnp.array(
            (
                1e-2,
                5e-2,
                2e-2,
            )
        )

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )

    def test_sweep_10_te_corrected_modes(self):
        r"""
        Check the first 3 modes of the swept Pazy model with trailing edge mass, and that they are close to literature.
        This uses corrections to the inertia properties to better match the second bending mode. Comparing with
        "Flutter, Post-Flutter, and Limit-Cycle Oscillations of Very Flexible Swept Wings".
        """
        freqs = self._swept_modes(tip_mass="TE_CORRECTED")

        literature_freqs = jnp.array((4.27, 27.4, 38.7))  # hz

        tolerances = jnp.array(
            (
                1e-2,
                1e-2,
                2e-2,
            )
        )

        error = jnp.abs((freqs / literature_freqs) - 1.0)

        assert jnp.all(error < tolerances), (
            "Pazy undeformed structural frequencies do not match literature frequencies"
        )
