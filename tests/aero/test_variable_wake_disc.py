from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot

from aero.uvlm import UVLM
from aero.utils import make_rectangular_grid
from aero.data_structures import GridDiscretization
from aero.flowfields import Constant
from print_utils import set_verbosity, VerbosityLevel


class TestVarWakeDisc:
    @staticmethod
    def test_var_wake_disc():
        set_verbosity(VerbosityLevel.SILENT)

        u_inf = jnp.array((10.0, 0.0, 0.0))
        rho_inf = 1.225
        m = 6
        n = 12
        m_star_base = 40
        m_star_var = 40
        c_ref = 1.0
        b_ref = 5.0
        alpha = jnp.deg2rad(0.0)
        ea = 0.0
        physical_time = 3.0  # seconds

        flowfield = Constant(u_inf, rho_inf, True)
        dt = c_ref / (m * flowfield.u_inf_mag)
        n_tstep = int(jnp.ceil(physical_time / dt))

        x_grid = make_rectangular_grid(m, n, c_ref, ea)

        beam_coords = jnp.zeros((n + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
        rmat = Rot.from_euler("xyz", jnp.array((0.0, alpha, 0.0))).as_matrix()

        # static position
        hg = jnp.zeros((n + 1, 4, 4))
        hg = hg.at[:, 3, 3].set(1.0)
        hg = hg.at[:, :3, :3].set(rmat[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        # heaving motion
        freq = 3.0  # Hz
        omega = 2 * jnp.pi * freq
        ampl = 0.1  # m
        t = jnp.arange(n_tstep) * dt
        z_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        z_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        hg_t = jnp.zeros((n_tstep, n + 1, 4, 4))
        hg_t = hg_t.at[:, 3, 3].set(1.0)
        hg_t = hg_t.at[...].set(hg[None, ...])
        hg_t = hg_t.at[:, :, 2, 3].add(z_t[:, None])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 2, 3].set(z_dot_t[:, None])

        # wake discretization for variable wake model
        delta_w_var = jnp.ones(m_star_var) * dt * flowfield.u_inf_mag

        gamma_b = []
        for delta_w_, m_star in [(delta_w_var, m_star_var), (None, m_star_base)]:
            variable_wake = delta_w_ is not None
            disc = GridDiscretization(m, n, m_star)

            uvlm = UVLM(
                grid_shapes=[disc],
                dof_mapping=jnp.arange(0, n + 1),
                variable_wake_disc=variable_wake,
            )

            uvlm.set_design_variables(dt, flowfield, delta_w_, x_grid, hg)

            static_case = uvlm.solve_static()

            dynamic_case = uvlm.solve_prescribed_dynamic(
                static_case, hg_t, hg_dot_t, False
            )
            gamma_b.append(dynamic_case.gamma_b[0])

        assert jnp.allclose(gamma_b[0], gamma_b[1], atol=1e-3), (
            "Heaving cantilever wing bound circulation strengths does not match between variable and base wake models."
        )
