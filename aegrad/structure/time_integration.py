from jax import Array, vmap
from jax import numpy as jnp

from aegrad.print_utils import warn
from algebra.se3 import log_se3, exp_se3
from structure.gradients.data_structures import UnsteadyStructureMinimalStates


class TimeIntregrator:
    r"""
    Container for time integration parameters.
    """

    def __init__(
        self,
        spectral_radius: float | Array,
        dt: float | Array,
    ):
        if 1.0 <= spectral_radius < 0.0:
            warn(
                "Spectral radius should be between 0.0 and 1.0 to guarantee stability."
            )
        self.dt: Array = jnp.array(dt)
        self.spectral_radius: Array = jnp.array(spectral_radius)
        self.alpha_m: Array = (2.0 * spectral_radius - 1.0) / (spectral_radius + 1.0)
        self.alpha_f: Array = spectral_radius / (spectral_radius + 1.0)
        self.gamma: Array = (3.0 - spectral_radius) / (2.0 + 2.0 * spectral_radius)
        self.beta: Array = 1.0 / ((spectral_radius + 1.0) ** 2)
        self.gamma_prime: Array = self.gamma / (self.beta * dt)
        self.beta_prime: Array = (1.0 - self.alpha_m) / (
            self.beta * dt * dt * (1.0 - self.alpha_f)
        )

    def predict_n(self, v_n: Array, a_n: Array, a_np1_pred: Array) -> Array:
        r"""
        Predict the next velocity based on current velocity and acceleration.
        :param v_n: Previous velocity, [n_nodes_, 6].
        :param a_n: Previous pseudoacceleration, [n_nodes_, 6].
        :param a_np1_pred: Next pseudoacceleration prediction [n_nodes_, 6].
        :return: Initial guess for next increment, [n_nodes_, 6].
        """
        return self.dt * v_n + self.dt * self.dt * (
            (0.5 - self.beta) * a_n + self.beta * a_np1_pred
        )

    def predict_v(self, v_n: Array, a_n: Array, a_np1_pred: Array) -> Array:
        r"""
        Predict the next velocity based on current velocity and acceleration.
        :param v_n: Previous velocity, [n_nodes_, 6].
        :param a_n: Previous pesudoacceleration, [n_nodes_, 6].
        :param a_np1_pred: Next pseudoacceleration prediction [n_nodes_, 6].
        :return: Initial guess for next velocity, [n_nodes_, 6].
        """
        return (
            v_n + (1.0 - self.gamma) * self.dt * a_n + self.gamma * self.dt * a_np1_pred
        )

    def predict_v_dot(self, v_dot_n: Array, a_n: Array, a_np1_pred: Array) -> Array:
        r"""
        Predict the acceleration based on previous pseudoacceleration and acceleration.
        :param v_dot_n: Previous acceleration, [n_nodes_, 6].
        :param a_n: Previous pseudoacceleration, [n_nodes_, 6].
        :param a_np1_pred: Next pseudoacceleration prediction [n_nodes_, 6].
        :return: Predicted acceleration, [n_nodes_, 6].
        """
        return (
            (1.0 - self.alpha_m) * a_np1_pred
            + self.alpha_m * a_n
            - self.alpha_f * v_dot_n
        ) / (1.0 - self.alpha_f)

    def predict_a(self, v_dot_n: Array, a_n: Array) -> Array:
        r"""
        Predict the pseudoacceleration based on previous pseudoacceleration and acceleration.
        :param v_dot_n: Previous acceleration, [n_nodes_, 6].
        :param a_n: Previous pseudoacceleration, [n_nodes_, 6].
        :return: Predicted pseudoacceleration, [n_nodes_, 6].
        """
        return (self.alpha_f * v_dot_n - self.alpha_m * a_n) / (1.0 - self.alpha_m)

    def calculate_a_np1(self, v_dot_n: Array, v_dot_np1: Array, a_n: Array) -> Array:
        r"""
        Calculate the pseudoacceleration at the next time step.
        :param v_dot_n: Previous acceleration, [n_nodes_, 6].
        :param v_dot_np1: Next acceleration, [n_nodes_, 6].
        :param a_n: Previous pseudoacceleration, [n_nodes_, 6].
        :return: Pseudoacceleration at next time step, [n_nodes_, 6].
        """
        return (
            1.0
            / (1.0 - self.alpha_m)
            * (
                (1.0 - self.alpha_f) * v_dot_np1
                + self.alpha_f * v_dot_n
                - self.alpha_m * a_n
            )
        )

    def calculate_f_alpha(self, f_n: Array, f_np1: Array) -> Array:
        return (1.0 - self.alpha_f) * f_np1 + self.alpha_f * f_n

    def calculate_phi_alpha(self, phi_np1: Array) -> Array:
        return (1.0 - self.alpha_f) * phi_np1

    def calculate_v_alpha(self, v_n: Array, v_np1: Array) -> Array:
        return (1.0 - self.alpha_f) * v_np1 + self.alpha_f * v_n

    def calculate_v_dot_alpha(self, v_dot_n: Array, v_dot_np1: Array) -> Array:
        return (1.0 - self.alpha_f) * v_dot_np1 + self.alpha_f * v_dot_n

    def calculate_a_alpha(self, a_n: Array, a_np1: Array) -> Array:
        return (1.0 - self.alpha_m) * a_np1 + self.alpha_m * a_n

    def calculate_varphi_alpha(self, varphi_n: Array, phi_np1: Array) -> Array:
        phi_alpha = self.calculate_phi_alpha(phi_np1)
        return vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), (0, 0), 0
        )(varphi_n, phi_alpha)

    def calculate_q_alpha(
        self, q_n: UnsteadyStructureMinimalStates, q_np1: UnsteadyStructureMinimalStates
    ) -> UnsteadyStructureMinimalStates:
        phi_alpha = self.calculate_phi_alpha(phi_np1=q_np1.phi)
        varphi_alpha = self.calculate_varphi_alpha(
            varphi_n=q_n.varphi, phi_np1=q_np1.phi
        )
        v_alpha = self.calculate_v_alpha(v_n=q_n.v, v_np1=q_np1.v)
        v_dot_alpha = self.calculate_v_dot_alpha(
            v_dot_n=q_n.v_dot, v_dot_np1=q_np1.v_dot
        )
        a_alpha = self.calculate_a_alpha(a_n=q_n.a, a_np1=q_np1.a)
        return UnsteadyStructureMinimalStates(
            phi=phi_alpha,
            varphi=varphi_alpha,
            v=v_alpha,
            v_dot=v_dot_alpha,
            a=a_alpha,
        )

    def calculate_phi_from_phi_alpha(self, phi_alpha: Array) -> Array:
        r"""
        Obtain the full timestep increment from the alpha increment.
        :param phi_alpha: Increment from timestep n-1 to alpha, [n_nodes, 6].
        :return: Increment for timestep n, [n_nodes, 6].
        """
        return phi_alpha / (1.0 - self.alpha_f)

    def calculate_v_from_v_alpha(self, v_alpha: Array, v_nm1: Array) -> Array:
        r"""
        Obtain the full timestep velocity from the alpha increment and the previous velocity.
        :param v_alpha: Velocity at alpha step, [n_nodes, 6].
        :param v_nm1: Velocity at timestep n-1, [n_nodes, 6].
        :return: Velocity at timestep n, [n_nodes, 6].
        """

        return (v_alpha - self.alpha_f * v_nm1) / (1.0 - self.alpha_f)

    def calculate_v_dot_from_v_dot_alpha(
        self, v_dot_alpha: Array, v_dot_nm1: Array
    ) -> Array:
        r"""
        Obtain the full timestep acceleration from the alpha increment and the previous acceleration.
        :param v_dot_alpha: Acceleration at alpha step, [n_nodes, 6].
        :param v_dot_nm1: Acceleration at timestep n-1, [n_nodes, 6].
        :return: Acceleration at timestep n, [n_nodes, 6].
        """

        return (v_dot_alpha - self.alpha_f * v_dot_nm1) / (1.0 - self.alpha_f)
