from jax import numpy as jnp
from jax import Array

from aegrad.aero.flowfields import OneMinusCosine
from aegrad.coupled.data_structures import AeroelasticFullStates
from aegrad.utils.data_structures import ConvergenceSettings

from models.cantilever_wing import make_cantilever_wing

# Discretisation parameters shared across all tests
u_inf = jnp.array((10.0, 0.0, 0.1))
k_cs_base = jnp.diag(jnp.array((1e2, 1e2, 1.0, 1.0, 1.0, 1.0)))
m_cs_base = jnp.diag(jnp.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
gust_amplitude_base = 0.3
dt = 0.2 / (2.0 * jnp.linalg.norm(u_inf))
n_tstep = 100


def _dynamic_objective(states: AeroelasticFullStates, *_, **__) -> Array:
    return states.structure.f_elem[0, 3] / n_tstep


def _build_wing(k_cs: Array, gust_amplitude: float | Array):
    wing = make_cantilever_wing(
        m=2,
        m_star=3,
        c_ref=0.2,
        b_ref=1.0,
        k_cs=k_cs,
        m_cs=m_cs_base,
        ea=0.25,
        n_nodes=5,
        u_inf=u_inf,
    )
    wing.aero.flowfield = OneMinusCosine(
        u_inf=u_inf,
        rho=1.225,
        relative_motion=True,
        gust_length=2.0,
        gust_amplitude=gust_amplitude,
        gust_x0=jnp.array((-5.0, 0.0, 0.0)),
    )
    conv = ConvergenceSettings(
        max_n_iter=100,
        rel_disp_tol=0.0,
        abs_disp_tol=0.0,
        rel_force_tol=0.0,
        abs_force_tol=0.0,
    )
    wing.structure.struct_convergence_settings = conv
    wing.fsi_convergence_settings = ConvergenceSettings(
        max_n_iter=40,
        rel_disp_tol=0.0,
        abs_disp_tol=0.0,
        rel_force_tol=0.0,
        abs_force_tol=0.0,
    )
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
        dt=dt,
        n_tstep=n_tstep,
        include_unsteady_aero_force=True,
    )
    return wing, static_sol, dynamic_sol


def _total_objective(wing, dynamic_sol) -> Array:
    """Sum the per-timestep objective over the full trajectory."""
    dv = wing.get_design_variables(case=dynamic_sol)
    total = jnp.array(0.0)
    for i in range(n_tstep):
        total += _dynamic_objective(dynamic_sol.get_full_states(i_ts=i), dv, i)
    return total


class TestDynamicGustAdjoint:
    @classmethod
    def setup_class(cls):
        cls.wing, cls.static_sol, cls.dynamic_sol = _run_primal(
            k_cs_base, gust_amplitude_base
        )
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
        _, _, sol_plus = _run_primal(k_cs_base, gust_amplitude_base + eps)
        _, _, sol_minus = _run_primal(k_cs_base, gust_amplitude_base - eps)
        wing_plus = _build_wing(k_cs_base, gust_amplitude_base + eps)
        wing_minus = _build_wing(k_cs_base, gust_amplitude_base - eps)

        fd_grad = (
            _total_objective(wing_plus, sol_plus)
            - _total_objective(wing_minus, sol_minus)
        ) / (2 * eps)
        adj_grad = self.dynamic_grad.aero.flowfield["gust_amplitude"].sum()

        assert jnp.allclose(fd_grad, adj_grad, rtol=1e-2), (
            f"Gradient mismatch w.r.t. gust_amplitude: adjoint={adj_grad:.6f}, FD={fd_grad:.6f}"
        )

    def test_k_cs_gradient(self):
        """Adjoint gradient w.r.t. k_cs[3, 3] verified by forward finite differences."""
        eps = 1e-3
        k_cs_plus = k_cs_base.at[3, 3].add(eps)
        wing_plus, _, sol_plus = _run_primal(k_cs_plus, gust_amplitude_base)

        fd_grad = (_total_objective(wing_plus, sol_plus) - self.baseline_obj) / eps
        if self.dynamic_grad.structure.k_cs is None:
            raise ValueError("k_cs grad is None")
        adj_grad = self.dynamic_grad.structure.k_cs[:, :, 3, 3].sum()

        assert jnp.allclose(fd_grad, adj_grad, rtol=1e-2), (
            f"Gradient mismatch w.r.t. k_cs[3,3]: adjoint={adj_grad:.6f}, FD={fd_grad:.6f}"
        )
