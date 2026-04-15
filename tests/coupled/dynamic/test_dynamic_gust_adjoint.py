from typing import Optional

from jax import numpy as jnp
from jax import Array
import jax

from aero.flowfields import OneMinusCosine
from coupled.gradients.data_structures import AeroelasticFullStates, AeroelasticDesignVariables
from utils.data_structures import ConvergenceSettings

from models.cantilever_wing import make_cantilever_wing

jax.config.update("jax_enable_x64", True)

# Discretisation parameters shared across all tests
_M = 2
_N = 4
_M_STAR = 3
_C_REF = 0.2
_B_REF = 1.0
_U_INF = jnp.array((10.0, 0.0, 0.1))
_U_INF_MAG = jnp.linalg.norm(_U_INF)
_K_CS_BASE = jnp.diag(jnp.array((1e2, 1e2, 1.0, 1.0, 1.0, 1.0)))
_M_CS_BASE = jnp.diag(jnp.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
_GUST_AMPLITUDE_BASE = 0.3
_DT = _C_REF / (_M * _U_INF_MAG)
_N_TSTEP = 100


def _dynamic_objective(states: AeroelasticFullStates, dv: AeroelasticDesignVariables,
                       i_ts: Optional[int | Array]) -> Array:
    return states.structure.f_elem[0, 3] / _N_TSTEP


def _build_wing(k_cs: Array, gust_amplitude: float | Array):
    wing = make_cantilever_wing(
        m=_M, m_star=_M_STAR, c_ref=_C_REF, b_ref=_B_REF,
        k_cs=k_cs, m_cs=_M_CS_BASE, ea=0.25, n_nodes=_N + 1, u_inf=_U_INF,
    )
    wing.aero.flowfield = OneMinusCosine(
        u_inf=_U_INF, rho=1.225, relative_motion=True,
        gust_length=2.0, gust_amplitude=gust_amplitude,
        gust_x0=jnp.array((-5.0, 0.0, 0.0)),
    )
    conv = ConvergenceSettings(max_n_iter=100, rel_disp_tol=0.0, abs_disp_tol=0.0,
                               rel_force_tol=0.0, abs_force_tol=0.0)
    wing.structure.struct_convergence_settings = conv
    wing.fsi_convergence_settings = ConvergenceSettings(max_n_iter=40, rel_disp_tol=0.0, abs_disp_tol=0.0,
                                                        rel_force_tol=0.0, abs_force_tol=0.0)
    return wing


def _run_primal(k_cs: Array, gust_amplitude: float | Array):
    wing = _build_wing(k_cs, gust_amplitude)
    static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))
    dynamic_sol = wing.dynamic_solve(
        init_case=static_sol,
        prescribed_dofs=jnp.arange(6),
        spectral_radius=1.0,
        gamma_dot_relaxation_factor=0.7,
        free_wake=False,
        dt=_DT,
        n_tstep=_N_TSTEP,
        include_unsteady_aero_force=True,
    )
    return wing, static_sol, dynamic_sol


def _total_objective(wing, dynamic_sol) -> Array:
    """Sum the per-timestep objective over the full trajectory."""
    dv = wing.get_design_variables(case=dynamic_sol)
    total = jnp.array(0.0)
    for i in range(_N_TSTEP):
        total += _dynamic_objective(dynamic_sol.get_full_states(i_ts=i), dv, i)
    return total


class TestDynamicGustAdjoint:
    @classmethod
    def setup_class(cls):
        cls.wing, cls.static_sol, cls.dynamic_sol = _run_primal(_K_CS_BASE, _GUST_AMPLITUDE_BASE)
        cls.baseline_obj = _total_objective(cls.wing, cls.dynamic_sol)

        # Forward static adjoint gives p_varphi/p_x (sensitivity of equilibrium states
        # to design variables), needed to propagate initial condition sensitivity for
        # parameters like k_cs that affect the static equilibrium.
        _, static_adj = cls.wing.static_adjoint(
            case=cls.static_sol, objective=_dynamic_objective, forward_adjoint=True
        )

        cls.dynamic_grad, _ = cls.wing.dynamic_adjoint(
            case=cls.dynamic_sol,
            objective=_dynamic_objective,
            p_varphi_p_x=-static_adj,
        )

    def test_gust_amplitude_gradient(self):
        """Adjoint gradient w.r.t. gust_amplitude verified by centered finite differences."""
        eps = 1e-3
        _, _, sol_plus = _run_primal(_K_CS_BASE, _GUST_AMPLITUDE_BASE + eps)
        _, _, sol_minus = _run_primal(_K_CS_BASE, _GUST_AMPLITUDE_BASE - eps)
        wing_plus = _build_wing(_K_CS_BASE, _GUST_AMPLITUDE_BASE + eps)
        wing_minus = _build_wing(_K_CS_BASE, _GUST_AMPLITUDE_BASE - eps)

        fd_grad = (_total_objective(wing_plus, sol_plus) - _total_objective(wing_minus, sol_minus)) / (2 * eps)
        adj_grad = self.dynamic_grad.aero.flowfield['gust_amplitude'].sum()

        assert jnp.allclose(fd_grad, adj_grad, rtol=1e-2), (
            f"Gradient mismatch w.r.t. gust_amplitude: adjoint={adj_grad:.6f}, FD={fd_grad:.6f}"
        )

    def test_k_cs_gradient(self):
        """Adjoint gradient w.r.t. k_cs[3, 3] verified by forward finite differences."""
        eps = 1e-3
        k_cs_plus = _K_CS_BASE.at[3, 3].add(eps)
        wing_plus, _, sol_plus = _run_primal(k_cs_plus, _GUST_AMPLITUDE_BASE)

        fd_grad = (_total_objective(wing_plus, sol_plus) - self.baseline_obj) / eps
        adj_grad = self.dynamic_grad.structure.k_cs[:, 3, 3].sum()

        assert jnp.allclose(fd_grad, adj_grad, rtol=1e-2), (
            f"Gradient mismatch w.r.t. k_cs[3,3]: adjoint={adj_grad:.6f}, FD={fd_grad:.6f}"
        )
