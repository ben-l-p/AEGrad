from jax import numpy as jnp

from aegrad.aero import make_rectangular_grid, GridDiscretization, UVLM
from aegrad.aero.flowfields import Constant
from aegrad.print_output import set_verbosity, VerbosityLevel


class TestRotInvariance:
    @staticmethod
    def test_rot_invariance_no_wake():
        r"""
        Test that the solution is rotation invariant for a no-wake case of a square wing, subject to flows in both
        the X and Y directions with positive angle of attack.
        """

        set_verbosity(VerbosityLevel.SILENT)
        mn = 5
        width = 1.0
        disc = GridDiscretization(mn, mn, 0)

        hg = jnp.zeros((mn + 1, 4, 4))
        x_grid = make_rectangular_grid(mn, mn, width, 0.0)
        beam_coords = jnp.zeros((mn + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, width, mn + 1))
        hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        case = UVLM(2, [disc], False, jnp.arange(0, mn + 1))

        for i_u_inf, u_inf in enumerate(
            [jnp.array((0.0, 10.0, 3.0)), jnp.array((10.0, 0.0, 3.0))]
        ):
            flowfield = Constant(u_inf, 1.225, True)
            case.set_design_variables(1.0, flowfield, None, x_grid, hg)
            case.solve_static(i_u_inf, None, False)

        if not jnp.allclose(case.gamma_b[0][0, ...], case.gamma_b[0][1, ...]):
            raise ValueError(
                "Gamma distribution is not equal for both flow directions in no-wake case."
            )

        if not jnp.allclose(
            f_tot := jnp.sum(case.f_steady[0][0, ...], axis=(0, 1)),
            0.0,
            atol=1e-5,
            rtol=1e-4,
        ):
            raise ValueError(f"Total force in flow is not zero: {f_tot}")

        if not jnp.allclose(
            case.f_steady[0][0, ...],
            jnp.transpose(case.f_steady[0][1, ...], (1, 0, 2))[..., (1, 0, 2)],
            atol=1e-5,
            rtol=1e-4,
        ):
            raise ValueError(
                "Steady force distribution is not equal in both flow directions in no-wake case."
            )
