from __future__ import annotations
from aegrad.aero.uvlm_utils import make_rectangular_grid
from aegrad.aero.data_structures import GridDiscretization, InputUnflattened
from aegrad.algebra.array_utils import ArrayList
from aegrad.aero.linear import LinearWakeType
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation as rot
from aegrad.aero.case import AeroCase
from aegrad.aero.flowfields import FlowField, Constant, OneMinusCosine
from aegrad.print_output import set_verbosity, VerbosityLevel
from pathlib import Path
from aegrad.aero.kernels import biot_savart_cutoff
from jax import Array, vmap


class TestLinearAero:
    u_inf: Array = jnp.array((10.0, 0.0, 1.0))
    rho_inf: float = 1.225

    @staticmethod
    def make_planar_wing(
        flowfield: FlowField,
        ea: float = 0.0,
    ) -> tuple[AeroCase, Array]:
        r"""
        Returns a reference wing case, and the reference beam coordinates.
        """
        m = 4
        n = 8
        m_star = 10
        c_ref = 1.0
        b_ref = 5.0
        alpha = jnp.deg2rad(0.0)
        physical_time = 1.0  # seconds

        dt = c_ref / (m * flowfield.u_inf_mag)
        n_tstep = int(jnp.ceil(physical_time / dt))
        disc = GridDiscretization(m, n, m_star)

        x_grid = make_rectangular_grid(m, n, c_ref, ea)

        beam_coords = jnp.zeros((n + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
        rmat = rot.from_euler("xyz", jnp.array((0.0, alpha, 0.0))).as_matrix()

        # static position
        hg = jnp.zeros((n + 1, 4, 4))
        hg = hg.at[:, 3, 3].set(1.0)
        hg = hg.at[:, :3, :3].set(rmat[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        # nonlinear case
        case = AeroCase(
            n_tstep, disc, False, jnp.arange(0, n + 1), kernel=biot_savart_cutoff
        )
        case.set_design_variables(dt, flowfield, None, x_grid, hg)
        case.solve_static()

        return case, hg

    @staticmethod
    def test_linear_operator_heaving_wing(plot: bool = False, use_matrix: bool = False):
        set_verbosity(VerbosityLevel.SILENT)

        flowfield = Constant(TestLinearAero.u_inf, TestLinearAero.rho_inf, True)
        case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = 0.1  # m
        t = jnp.arange(case.n_tstep_tot) * case.dt
        z_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        z_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        hg_t = jnp.zeros((case.n_tstep_tot, case.grid_disc[0].n + 1, 4, 4))
        hg_t = hg_t.at[:, 3, 3].set(1.0)
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, 2, 3].add(z_t[:, None])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 2, 3].set(z_dot_t[:, None])

        # nonlinear case
        case.solve_static()
        case.solve_prescribed_dynamic(hg_t, hg_dot_t, False)
        if plot:
            case.plot(Path("./test_outputs/heaving_test_nonlinear"))

        # linear case
        linear_model = case.linearise(
            0,
            LinearWakeType.PRESCRIBED,
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.zeta0_b]
        )
        u_linear = InputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_model.run(u_linear, use_matrix=use_matrix)

        if plot:
            case.plot(
                Path(
                    f"./test_outputs/heaving_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            case.gamma_b[0], linear_model.x_t_tot.gamma_b[0], atol=5e-3
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.gamma_w[0], linear_model.x_t_tot.gamma_w[0], atol=5e-3
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.zeta_w[0], linear_model.x_t_tot.zeta_w[0], atol=1e-3
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_steady[0], linear_model.y_t_tot.f_steady[0], atol=1e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_unsteady[0],
            linear_model.y_t_tot.f_unsteady[0],
            atol=2e-2,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @staticmethod
    def test_linear_matrix_heaving_wing(plot: bool = False):
        TestLinearAero.test_linear_operator_heaving_wing(plot=plot, use_matrix=True)

    @staticmethod
    def test_linear_operator_pitching_wing(
        plot: bool = False, use_matrix: bool = False
    ):
        set_verbosity(VerbosityLevel.SILENT)

        flowfield = Constant(TestLinearAero.u_inf, TestLinearAero.rho_inf, True)
        case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = jnp.deg2rad(4.0)  # deg
        t = jnp.arange(case.n_tstep_tot) * case.dt
        alpha_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        alpha_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        rmat_t = vmap(
            lambda angle: rot.from_euler(
                "xyz", jnp.array((0.0, angle, 0.0))
            ).as_matrix()
        )(alpha_t)

        hg_t = jnp.zeros((case.n_tstep_tot, case.grid_disc[0].n + 1, 4, 4))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, :3, :3].set(rmat_t[:, None, :, :])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 0, 2].set(alpha_dot_t[:, None])
        hg_dot_t = hg_dot_t.at[:, :, 2, 0].set(-alpha_dot_t[:, None])

        # nonlinear case
        case.solve_static()
        case.solve_prescribed_dynamic(hg_t, hg_dot_t, False)
        if plot:
            case.plot(Path("./test_outputs/pitching_test_nonlinear"))

        # linear case
        linear_model = case.linearise(
            0,
            LinearWakeType.PRESCRIBED,
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.zeta0_b]
        )
        u_linear = InputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_model.run(u_linear, use_matrix=use_matrix)

        if plot:
            case.plot(
                Path(
                    f"./test_outputs/pitching_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            case.gamma_b[0], linear_model.x_t_tot.gamma_b[0], atol=4e-2
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.gamma_w[0], linear_model.x_t_tot.gamma_w[0], atol=4e-2
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.zeta_w[0], linear_model.x_t_tot.zeta_w[0], atol=1e-3
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_steady[0], linear_model.y_t_tot.f_steady[0], atol=3e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_unsteady[0],
            linear_model.y_t_tot.f_unsteady[0],
            atol=1e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @staticmethod
    def test_linear_matrix_pitching_wing(plot: bool = False):
        TestLinearAero.test_linear_operator_pitching_wing(plot=plot, use_matrix=True)

    @staticmethod
    def test_linear_operator_pitching_wing_frozen_wake(
        plot: bool = False, use_matrix: bool = False
    ):
        set_verbosity(VerbosityLevel.SILENT)

        flowfield = Constant(TestLinearAero.u_inf, TestLinearAero.rho_inf, True)
        case, hg0 = TestLinearAero.make_planar_wing(flowfield, ea=1.0)

        # heaving motion
        freq = 3.0  # Hz
        omega = 0.5 * jnp.pi * freq
        ampl = jnp.deg2rad(4.0)  # deg
        t = jnp.arange(case.n_tstep_tot) * case.dt
        alpha_t = ampl * 0.5 * (1.0 - jnp.cos(omega * t))
        alpha_dot_t = ampl * omega * 0.5 * jnp.sin(omega * t)

        rmat_t = vmap(
            lambda angle: rot.from_euler(
                "xyz", jnp.array((0.0, angle, 0.0))
            ).as_matrix()
        )(alpha_t)

        hg_t = jnp.zeros((case.n_tstep_tot, case.grid_disc[0].n + 1, 4, 4))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_t = hg_t.at[:, :, :3, :3].set(rmat_t[:, None, :, :])

        hg_dot_t = jnp.zeros_like(hg_t)
        hg_dot_t = hg_dot_t.at[:, :, 0, 2].set(alpha_dot_t[:, None])
        hg_dot_t = hg_dot_t.at[:, :, 2, 0].set(-alpha_dot_t[:, None])

        # nonlinear case
        case.solve_static()
        case.solve_prescribed_dynamic(hg_t, hg_dot_t, False)
        if plot:
            case.plot(Path("./test_outputs/pitching_test_nonlinear_frozen_wake"))

        # linear case
        linear_model = case.linearise(
            0,
            LinearWakeType.FROZEN,
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )

        delta_zeta_b = case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.zeta0_b]
        )
        u_linear = InputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_model.run(u_linear, use_matrix=use_matrix)

        if plot:
            case.plot(
                Path(
                    f"./test_outputs/pitching_test_linear_frozen_wake_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            case.gamma_b[0], linear_model.x_t_tot.gamma_b[0], atol=4e-2
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.gamma_w[0], linear_model.x_t_tot.gamma_w[0], atol=4e-2
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_steady[0], linear_model.y_t_tot.f_steady[0], atol=3e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_unsteady[0],
            linear_model.y_t_tot.f_unsteady[0],
            atol=1e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @staticmethod
    def test_linear_matrix_pitching_wing_frozen_wake(plot: bool = False):
        TestLinearAero.test_linear_operator_pitching_wing_frozen_wake(
            plot=plot, use_matrix=True
        )

    @staticmethod
    def test_linear_operator_cosine_gust(plot: bool = False, use_matrix: bool = False):
        set_verbosity(VerbosityLevel.SILENT)

        flowfield = OneMinusCosine(
            TestLinearAero.u_inf,
            TestLinearAero.rho_inf,
            True,
            gust_length=2.0,
            gust_amplitude=0.4,
            gust_x0=jnp.array((-5.0, 0.0, 0.0)),
        )
        case, hg0 = TestLinearAero.make_planar_wing(flowfield)

        hg_t = jnp.zeros((case.n_tstep_tot, *hg0.shape))
        hg_t = hg_t.at[...].set(hg0[None, ...])
        hg_dot_t = jnp.zeros_like(hg_t)

        # nonlinear case
        case.solve_static()
        case.solve_prescribed_dynamic(hg_t, hg_dot_t, False)
        if plot:
            case.plot(Path("./test_outputs/gust_test_nonlinear"))

        # linear case
        linear_model = case.linearise(
            0,
            LinearWakeType.PRESCRIBED,
            bound_upwash=True,
            wake_upwash=True,
            unsteady_force=True,
        )

        delta_zeta_b = case.zeta_b - ArrayList(
            [zeta[None, ...] for zeta in linear_model.zeta0_b]
        )
        u_linear = InputUnflattened(
            zeta_b=delta_zeta_b,
            zeta_b_dot=case.zeta_b_dot,
            nu_b=None,
            nu_w=None,
        )

        # Run linear case with and without matrix form
        linear_model.run(u_linear, use_matrix=use_matrix, flowfield=flowfield)

        if plot:
            case.plot(
                Path(
                    f"./test_outputs/gust_test_linear_{'matrix' if use_matrix else 'operator'}"
                )
            )

        assert jnp.allclose(
            case.gamma_b[0], linear_model.x_t_tot.gamma_b[0], atol=3e-1
        ), "Bound circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.gamma_w[0], linear_model.x_t_tot.gamma_w[0], atol=3e-1
        ), "Wake circulation does not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.zeta_w[0], linear_model.x_t_tot.zeta_w[0], atol=8e-2
        ), "Wake grid coordinates do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_steady[0], linear_model.y_t_tot.f_steady[0], atol=6e-1
        ), "Steady forces do not match between nonlinear and linear cases."
        assert jnp.allclose(
            case.f_unsteady[0],
            linear_model.y_t_tot.f_unsteady[0],
            atol=8e-1,
        ), "Unsteady forces do not match between nonlinear and linear cases."

    @staticmethod
    def test_linear_matrix_cosine_gust(plot: bool = False):
        TestLinearAero.test_linear_operator_cosine_gust(plot=plot, use_matrix=True)
