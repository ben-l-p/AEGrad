import pytest
from jax import Array
from jax import numpy as jnp

from aegrad.aero.flowfields import Constant
from aegrad.algebra.so3 import vec_to_skew
from aegrad.models.pazy.straight.pazy_wing import make_pazy_wing
from aegrad.models.pazy.swept.swept_pazy_wing import make_swept_pazy_wing


def generate_straight_static_aeroelastic(
    aoa: float | Array, q_inf: float | Array
) -> Array:
    r"""
    Function to perform a static aeroelastic analysis of the Pazy wing and return the tip z-displacement.
    """
    rho = 1.225
    u_inf_mag = jnp.sqrt(2 * q_inf / rho)

    wing_ = make_pazy_wing(
        gravity=False,
        node_multiplier=2,
        m=16,
        m_star=160,
        skin=True,
        aoa=aoa,
        flowfield=Constant(
            rho=rho, u_inf=jnp.array((u_inf_mag, 0.0, 0.0)), relative_motion=True
        ),
    )
    static_sol = wing_.static_solve(
        prescribed_dofs=jnp.arange(6),
        horseshoe=False,
    )

    return 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )


def generate_swept_static_aeroelastic(
    aoa: float | Array, q_inf: float | Array
) -> Array:
    r"""
    Function to perform a static aeroelastic analysis of the swept Pazy wing and return the tip z-displacement.
    """
    rho = 1.225
    u_inf_mag = jnp.sqrt(2 * q_inf / rho)

    wing_ = make_swept_pazy_wing(
        gravity=False,
        node_multiplier=2,
        m=16,
        m_star=160,
        tip_mass="TE_CORRECTED",
        aoa=aoa,
        flowfield=Constant(
            rho=rho, u_inf=jnp.array((u_inf_mag, 0.0, 0.0)), relative_motion=True
        ),
    )
    static_sol = wing_.static_solve(
        prescribed_dofs=jnp.arange(6),
        horseshoe=False,
    )

    # rotating a swept wing gives a non-zero reference z
    reference_sol = wing_.reference_configuration(prescribed_dofs=tuple(range(6)))
    reference_tip_z = 0.5 * (
        reference_sol.aero.zeta_b[0][0, -1, 2] + reference_sol.aero.zeta_b[0][-1, -1, 2]
    )

    tip_z = 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )
    return (tip_z - reference_tip_z) / 0.55


class TestPazy:
    @classmethod
    def test_tip_z_force(cls):
        r"""
        Compare the Pazy wing tip z-displacement under a vertical dead load. Use a point mass to create load to
        match SHARPy case.
        """

        wing = make_pazy_wing(
            m=16,
            m_star=160,
            skin=True,
            node_multiplier=4,
            gravity=jnp.array((0.0, 0.0, -9.81)),
            aoa=0.0,
            model="prepazy",
        )
        wing.structure.use_lumped_mass = True
        wing.structure.m_lumped_index = (-1,)

        # mass of 3.5 kg (used as SHARPy doesn't allow directly adding a force to the tip node)
        m_tip = 3.5
        m_lumped = jnp.zeros((6, 6))
        m_lumped = m_lumped.at[:3, :3].set(m_tip * jnp.eye(3))

        wing.structure.m_lumped = m_lumped[None, ...]

        static_beam = wing.structure.static_solve(prescribed_dofs=jnp.arange(6))
        z_tip = static_beam.x[-1, 2]
        z_tip_sharpy = -0.283929
        assert jnp.isclose(z_tip, z_tip_sharpy, rtol=2e-5)

    @classmethod
    def test_tip_torsion(cls):
        r"""
        Compare the Pazy wing tip deflection under a tip torsion load. Use a point mass to create load to match SHARPy
        case.
        """

        wing = make_pazy_wing(
            m=16,
            m_star=160,
            skin=True,
            node_multiplier=4,
            gravity=jnp.array((0.0, 0.0, -9.81)),
            aoa=0.0,
            model="prepazy",
        )
        wing.structure.use_lumped_mass = True
        wing.structure.m_lumped_index = (-1,)

        # mass of 3.5 kg (used as SHARPy doesn't allow directly adding a force to the tip node)
        m_tip = 3.5
        l_tip = 0.0441 + 0.3  # 30 cm ahead of the leading edge
        m_lumped = jnp.zeros((6, 6))
        m_lumped = m_lumped.at[:3, :3].set(m_tip * jnp.eye(3))

        ml_skew = vec_to_skew(jnp.array([-l_tip * m_tip, 0.0, 0.0]))
        m_lumped = m_lumped.at[:3, 3:].set(-ml_skew)
        m_lumped = m_lumped.at[3:, :3].set(ml_skew)

        wing.structure.m_lumped = m_lumped[None, ...]

        static_beam = wing.structure.static_solve(
            prescribed_dofs=jnp.arange(6), load_steps=10
        )

        tip_coord = static_beam.x[-1, :]
        tip_coord_sharpy = jnp.array([0.0358353, 0.480385, -0.241945])

        assert jnp.allclose(tip_coord, tip_coord_sharpy, rtol=2e-4)

    @staticmethod
    @pytest.mark.parametrize(
        "aoa_deg, q_inf, z_tip_bounds",
        [
            pytest.param(3.0, 1200.0, (0.075, 0.09), id="aoa=3, q_inf=1200"),
            pytest.param(3.0, 2200.0, (0.16, 0.18), id="aoa=3, q_inf=2200"),
            pytest.param(5.0, 1200.0, (0.12, 0.14), id="aoa=5, q_inf=1200"),
            pytest.param(5.0, 2200.0, (0.23, 0.25), id="aoa=5, q_inf=2200"),
            pytest.param(7.0, 1200.0, (0.16, 0.185), id="aoa=7, q_inf=1200"),
            pytest.param(7.0, 2200.0, (0.27, 0.31), id="aoa=7, q_inf=2200"),
        ],
    )
    def test_static_aeroelastic(
        aoa_deg: float, q_inf: float, z_tip_bounds: tuple[float, float]
    ):
        r"""
        Test the static aeroelastic analysis of the Pazy wing across a sweep of angles of attack and dynamic pressures.
        Bounds are chosen to ensure the results are aligned with that found in "Collaborative Pazy Wing Analyses for
        the Third Aeroelastic Prediction Workshop", DOI: 10.2514/6.2024-0419.
        """
        z_tip = generate_straight_static_aeroelastic(
            aoa=jnp.deg2rad(aoa_deg), q_inf=q_inf
        )

        assert z_tip_bounds[0] < z_tip < z_tip_bounds[1], (
            f"Pazy vertical tip displacement {float(z_tip)} out of bounds: {z_tip_bounds}"
        )


class TestSwept10Pazy:
    @staticmethod
    @pytest.mark.parametrize(
        "aoa_deg, q_inf, z_tip_bounds",
        [
            pytest.param(3.0, 1000.0, (0.04, 0.09), id="aoa=3, q_inf=1000"),
            pytest.param(3.0, 3000.0, (0.15, 0.18), id="aoa=3, q_inf=3000"),
            pytest.param(5.0, 1000.0, (0.10, 0.15), id="aoa=5, q_inf=1000"),
            pytest.param(5.0, 3000.0, (0.25, 0.32), id="aoa=5, q_inf=3000"),
            pytest.param(7.0, 1000.0, (0.14, 0.19), id="aoa=7, q_inf=1000"),
            pytest.param(7.0, 3000.0, (0.35, 0.40), id="aoa=7, q_inf=3000"),
        ],
    )
    def test_static_aeroelastic(
        aoa_deg: float, q_inf: float, z_tip_bounds: tuple[float, float]
    ):
        r"""
        Test the static aeroelastic analysis of the 10-degree swept Pazy wing across a sweep of angles of attack and
        dynamic pressures. Bounds are chosen to ensure the results are aligned with that found in "Flutter, Post-Flutter, and
        Limit-Cycle Oscillations of Very Flexible Swept Wings". Note that the tolerances are quite relaxed as it is
        accepted that the results do not perfectly align, with the tests being to ensure a correct order of magnitude.
        Note that data for the swept case is normalised whereas it is not for the straight case.
        """
        z_tip = generate_swept_static_aeroelastic(aoa=jnp.deg2rad(aoa_deg), q_inf=q_inf)

        assert z_tip_bounds[0] < z_tip < z_tip_bounds[1], (
            f"Pazy vertical tip displacement {float(z_tip)} out of bounds: {z_tip_bounds}"
        )
