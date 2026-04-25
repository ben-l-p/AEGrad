from typing import Optional

from jax import numpy as jnp
from jax import Array

from aegrad.coupled.data_structures import (
    AeroelasticFullStates,
    AeroelasticDesignVariables,
)
from aegrad.utils.data_structures import ConvergenceSettings

from models.cantilever_wing import make_cantilever_wing


class TestDynamicEquilibriumAdjoint:
    def test_dynamic_equilibrium_adjoint(self, plot: bool = False):
        m = 2
        n = 4
        m_star = 3
        c_ref = 0.2
        b_ref = 1.0
        u_inf = jnp.array((10.0, 0.0, 0.1))
        u_inf_mag = jnp.linalg.norm(u_inf)
        k_cs = jnp.diag(jnp.array((1e2, 1e2, 1.0, 1.0, 1.0, 1.0)))

        wing = make_cantilever_wing(
            m=m,
            m_star=m_star,
            c_ref=c_ref,
            b_ref=b_ref,
            k_cs=k_cs,
            ea=0.25,
            n_nodes=n + 1,
            u_inf=u_inf,
        )

        dt = c_ref / (m * u_inf_mag)
        n_tstep = 100

        def static_objective(states: AeroelasticFullStates, *_, **__) -> Array:
            return states.structure.f_elem[0, 3]

        def dynamic_objective(
            states: AeroelasticFullStates,
            dv: AeroelasticDesignVariables,
            i_ts: Optional[int | Array],
        ) -> Array:
            return static_objective(states, dv, i_ts=i_ts) / n_tstep

        # set tolerance to zero, rather than none, to prevent error messages
        wing.structure.struct_convergence_settings = ConvergenceSettings(
            max_n_iter=100,
            rel_disp_tol=0.0,
            abs_disp_tol=0.0,
            rel_force_tol=0.0,
            abs_force_tol=0.0,
        )
        wing.fsi_convergence_settings = ConvergenceSettings(
            max_n_iter=40,
            rel_disp_tol=0.0,
            abs_disp_tol=0.0,
            rel_force_tol=0.0,
            abs_force_tol=0.0,
        )

        static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))

        dynamic_sol = wing.dynamic_solve(
            init_case=static_sol,
            prescribed_dofs=jnp.arange(6),
            spectral_radius=1.0,
            gamma_dot_relaxation_factor=0.7,
            free_wake=False,
            dt=dt,
            n_tstep=n_tstep,
            include_unsteady_aero_force=False,
        )

        if plot:
            dynamic_sol.plot(directory="./test_outputs/dynamic_coupled_adjoint")

        static_grad, static_adj = wing.static_adjoint(
            case=static_sol, objective=static_objective, forward_adjoint=True
        )

        dynamic_grad, dynamic_adj = wing.dynamic_adjoint(
            case=dynamic_sol, objective=dynamic_objective, p_varphi_p_x=-static_adj
        )

        assert jnp.allclose(
            dynamic_grad.aero.flowfield["u_inf"], static_grad.aero.flowfield["u_inf"]
        ), "Mismatch in u_inf gradient"

        assert jnp.allclose(
            dynamic_grad.aero.flowfield["rho"], static_grad.aero.flowfield["rho"]
        ), "Mismatch in rho gradient"

        if dynamic_grad.structure.m_cs is None:
            raise ValueError("Missing mass gradient")
        assert jnp.allclose(dynamic_grad.structure.m_cs, 0.0, atol=1e-6), (
            "Nonzero mass gradient"
        )

        if dynamic_grad.structure.k_cs is None:
            raise ValueError("Missing dynamic stiffness gradient")
        if static_grad.structure.k_cs is None:
            raise ValueError("Missing static stiffness gradient")
        assert jnp.allclose(dynamic_grad.structure.k_cs, static_grad.structure.k_cs), (
            "Mismatch in stiffness gradient"
        )
