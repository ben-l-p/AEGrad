from jax import numpy as jnp
import jax

from data_structures import ConvergenceSettings
from models.cantilever_wing import make_cantilever_wing

jax.config.update("jax_enable_x64", True)


class TestDynamicEquilibrium:
    def test_dynamic_equilibrium(self, plot: bool = False):
        r"""
        Ensure that a cantilever wing starting in its static equilibrium remains there when time stepped
        """

        m = 8
        n = 20
        m_star = 20
        c_ref = 1.0
        u_inf = jnp.array((10.0, 0.0, 1.5))
        u_inf_mag = jnp.linalg.norm(u_inf)
        k_cs = jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e3, 1e3, 1e3)))

        wing = make_cantilever_wing(m=m, m_star=m_star, c_ref=c_ref, k_cs=k_cs, ea=0.25, n_nodes=n + 1, u_inf=u_inf)

        # strict convergence
        conv_settings = ConvergenceSettings(max_n_iter=25, abs_disp_tol=1e-9, rel_disp_tol=1e-7, abs_force_tol=1e-9,
                                            rel_force_tol=1e-7)
        wing.structure.struct_convergence_settings = conv_settings
        wing.fsi_convergence_settings = conv_settings

        dt = c_ref / (m * u_inf_mag)
        n_tstep = 100

        static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))

        dynamic_sol = wing.dynamic_solve(init_case=static_sol, prescribed_dofs=jnp.arange(6), spectral_radius=0.8,
                                         free_wake=False, dt=dt, n_tstep=n_tstep, include_unsteady_aero_force=True)

        if plot:
            dynamic_sol.plot(directory=f"./test_outputs/dynamic_equilibrium")

        gamma_b_err = dynamic_sol.aero.gamma_b[0] - dynamic_sol.aero.gamma_b[0][[0], ...]

        assert jnp.allclose(jnp.abs(gamma_b_err).max(), 0.0, atol=6e-7), "Bound circulation varies from equilibrium"

        varphi_diff = dynamic_sol.structure.varphi - dynamic_sol.structure.varphi[[0], ...]

        assert jnp.allclose(jnp.abs(varphi_diff).max(), 0.0,
                            atol=6e-7), "Structural deformation varies from equilibrium"
