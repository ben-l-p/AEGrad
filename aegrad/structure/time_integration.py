from jax import Array
from jax import numpy as jnp

from aegrad.print_utils import warn


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
