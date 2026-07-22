import pytest
from jax import Array
from jax import numpy as jnp

from aegrad.aero.flowfields import Constant
from aegrad.algebra.so3 import vec_to_skew
from models.straight_pazy.pazy_wing import make_pazy_wing


def generate_static_aeroelastic(aoa: float | Array, q_inf: float | Array) -> Array:
    r"""
    Function to perform a static aeroelastic analysis of the Pazy wing and return the tip z-displacement.
    """
    rho = 1.225
    u_inf_mag = jnp.sqrt(2 * q_inf / rho)

    wing_ = make_pazy_wing(
        gravity=True,
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
        load_steps=1,
        fsi_relaxation=0.5,
    )

    return 0.5 * (
        static_sol.aero.zeta_b[0][0, -1, 2] + static_sol.aero.zeta_b[0][-1, -1, 2]
    )


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
        z_tip = generate_static_aeroelastic(aoa=jnp.deg2rad(aoa_deg), q_inf=q_inf)

        assert z_tip_bounds[0] < z_tip < z_tip_bounds[1], (
            f"Pazy vertical tip displacement {float(z_tip)} out of bounds: {z_tip_bounds}"
        )
