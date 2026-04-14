from jax import Array, vmap
from jax import numpy as jnp

from print_utils import warn
from algebra.se3 import log_se3, exp_se3, hg_to_d
from structure.data_structures import StructureMinimalStates


class TimeIntegrator:
    r"""
    Container for time integration parameters.
    """

    def __init__(
            self,
            spectral_radius: float | Array,
            dt: float | Array,
    ):
        if 1.0 < spectral_radius < 0.0:
            warn(
                "Spectral radius should be between 0.0 and 1.0 to guarantee stability."
            )
        spectral_radius: Array = jnp.array(spectral_radius)
        self.dt: Array = jnp.array(dt)
        self.spectral_radius: Array = spectral_radius
        self.alpha_m: Array = (2.0 * spectral_radius - 1.0) / (spectral_radius + 1.0)
        self.alpha_f: Array = spectral_radius / (spectral_radius + 1.0)
        self.gamma: Array = (3.0 - spectral_radius) / (2.0 + 2.0 * spectral_radius)
        self.beta: Array = 1.0 / ((spectral_radius + 1.0) ** 2)
        self.gamma_prime: Array = self.gamma / (self.beta * dt)
        self.beta_prime: Array = (1.0 - self.alpha_m) / (
                self.beta * dt * dt * (1.0 - self.alpha_f)
        )

    def calculate_a_n(self, v_dot_nm1: Array, v_dot_n: Array, a_nm1: Array) -> Array:
        r"""
        Calculate the pseudo-acceleration at the next time step.
        :param v_dot_nm1: Previous acceleration, [n_nodes_, 6].
        :param v_dot_n: Next acceleration, [n_nodes_, 6].
        :param a_nm1: Previous pseudo-acceleration, [n_nodes_, 6].
        :return: pseudo-acceleration at next time step, [n_nodes_, 6].
        """
        return (
                1.0
                / (1.0 - self.alpha_m)
                * (
                        (1.0 - self.alpha_f) * v_dot_n
                        + self.alpha_f * v_dot_nm1
                        - self.alpha_m * a_nm1
                )
        )

    def calculate_q_n_from_q_alpha(
            self,
            q_nm1: StructureMinimalStates,
            q_alpha: StructureMinimalStates,
            phi_alpha: Array,
    ) -> tuple[StructureMinimalStates, Array]:
        phi = phi_alpha / (1.0 - self.alpha_f)
        varphi = vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), 0, 0
        )(q_nm1.varphi, phi)
        v = (q_alpha.v - self.alpha_f * q_nm1.v) / (1.0 - self.alpha_f)
        v_dot = (q_alpha.v_dot - self.alpha_f * q_nm1.v_dot) / (1.0 - self.alpha_f)
        a = (q_alpha.a - self.alpha_m * q_nm1.a) / (1.0 - self.alpha_m)
        return StructureMinimalStates(varphi=varphi, v=v, v_dot=v_dot, a=a), phi

    def predict_q(
            self, q_nm1: StructureMinimalStates
    ) -> tuple[Array, StructureMinimalStates]:
        r"""
        Predict the current state based upon the previous state.
        :param q_nm1: State at timestep varphi
        :return: Predicted state at timestep varphi+1
        """
        a_n = (self.alpha_f * q_nm1.v_dot - self.alpha_m * q_nm1.a) / (1.0 - self.alpha_m)

        phi_n = self.dt * q_nm1.v + self.dt * self.dt * ((0.5 - self.beta) * q_nm1.a + self.beta * a_n)

        varphi_n = vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), 0, 0
        )(q_nm1.varphi, phi_n)

        v_n = q_nm1.v + (1.0 - self.gamma) * self.dt * q_nm1.a + self.gamma * self.dt * a_n

        v_dot_n = ((1.0 - self.alpha_m) * a_n + self.alpha_m * q_nm1.a - self.alpha_f * q_nm1.v_dot) / (
                1.0 - self.alpha_f)

        return phi_n, StructureMinimalStates(varphi=varphi_n, v=v_n, v_dot=v_dot_n, a=a_n)

    def calculate_f_alpha(self, f_nm1: Array, f_n: Array) -> Array:
        return (1.0 - self.alpha_f) * f_n + self.alpha_f * f_nm1

    def calculate_phi_alpha(self, phi_n: Array) -> Array:
        return (1.0 - self.alpha_f) * phi_n

    def calculate_v_alpha(self, v_nm1: Array, v_n: Array) -> Array:
        return (1.0 - self.alpha_f) * v_n + self.alpha_f * v_nm1

    def calculate_v_dot_alpha(self, v_dot_nm1: Array, v_dot_n: Array) -> Array:
        return (1.0 - self.alpha_f) * v_dot_n + self.alpha_f * v_dot_nm1

    def calculate_a_alpha(self, a_nm1: Array, a_n: Array) -> Array:
        return (1.0 - self.alpha_m) * a_n + self.alpha_m * a_nm1

    def calculate_varphi_alpha(self, varphi_nm1: Array, varphi_n: Array) -> Array:
        phi_n = vmap(lambda a, b: hg_to_d(exp_se3(a), exp_se3(b)))(varphi_nm1, varphi_n)

        phi_alpha = self.calculate_phi_alpha(phi_n)
        return vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), (0, 0), 0
        )(varphi_nm1, phi_alpha)

    def calculate_q_alpha(
            self, q_nm1: StructureMinimalStates, q_n: StructureMinimalStates, phi_n: Array
    ) -> tuple[Array, StructureMinimalStates]:
        phi_alpha = self.calculate_phi_alpha(phi_n)
        varphi_alpha = self.calculate_varphi_alpha(
            varphi_nm1=q_nm1.varphi, varphi_n=q_n.varphi
        )
        v_alpha = self.calculate_v_alpha(v_nm1=q_nm1.v, v_n=q_n.v)
        v_dot_alpha = self.calculate_v_dot_alpha(
            v_dot_nm1=q_nm1.v_dot, v_dot_n=q_n.v_dot
        )
        a_alpha = self.calculate_a_alpha(a_nm1=q_nm1.a, a_n=q_n.a)
        return phi_alpha, StructureMinimalStates(
            varphi=varphi_alpha,
            v=v_alpha,
            v_dot=v_dot_alpha,
            a=a_alpha,
        )

    def calculate_phi_from_phi_alpha(self, phi_alpha: Array) -> Array:
        r"""
        Obtain the full timestep increment from the alpha increment.
        :param phi_alpha: Increment from timestep varphi-1 to alpha, [n_nodes, 6].
        :return: Increment for timestep varphi, [n_nodes, 6].
        """
        return phi_alpha / (1.0 - self.alpha_f)

    def calculate_v_from_v_alpha(self, v_alpha: Array, v_nm1: Array) -> Array:
        r"""
        Obtain the full timestep velocity from the alpha increment and the previous velocity.
        :param v_alpha: Velocity at alpha step, [n_nodes, 6].
        :param v_nm1: Velocity at timestep varphi-1, [n_nodes, 6].
        :return: Velocity at timestep varphi, [n_nodes, 6].
        """

        return (v_alpha - self.alpha_f * v_nm1) / (1.0 - self.alpha_f)
