from jax import Array, vmap
from jax import numpy as jnp

from aegrad.print_utils import warn
from algebra.se3 import log_se3, exp_se3
from structure.data_structures import StructureMinimalStates


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

    def predict_phi(self, v_n: Array, a_n: Array, a_np1_pred: Array) -> Array:
        r"""
        Predict the next increment based on current velocity and acceleration.
        :param v_n: Previous velocity, [n_nodes_, 6].
        :param a_n: Previous pseudoacceleration, [n_nodes_, 6].
        :param a_np1_pred: Next pseudoacceleration prediction [n_nodes_, 6].
        :return: Initial guess for next increment, [n_nodes_, 6].
        """
        return self.dt * v_n + self.dt * self.dt * (
            (0.5 - self.beta) * a_n + self.beta * a_np1_pred
        )

    @staticmethod
    def predict_varphi(varphi_n: Array, phi_np1_pred: Array) -> Array:
        r"""
        Predict the next total twist based on current velocity and acceleration.
        :param varphi_n: Previous total twist, [n_nodes_, 6].
        :param phi_np1_pred: Predicted step increment, [n_nodes_, 6].
        :return: Initial guess for next twist, [n_nodes_, 6].
        """
        return vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), 0, 0
        )(varphi_n, phi_np1_pred)

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

    def calculate_q_n_from_q_alpha(
        self, q_nm1: StructureMinimalStates, q_alpha: StructureMinimalStates
    ) -> StructureMinimalStates:
        phi = q_alpha.phi / (1.0 - self.alpha_f)
        varphi = vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), 0, 0
        )(q_nm1.varphi, phi)
        v = (q_alpha.v - self.alpha_f * q_nm1.v) / (1.0 - self.alpha_f)
        v_dot = (q_alpha.v_dot - self.alpha_f * q_nm1.v_dot) / (1.0 - self.alpha_f)
        a = (q_alpha.a - self.alpha_m * q_nm1.a) / (1.0 - self.alpha_m)
        return StructureMinimalStates(phi=phi, varphi=varphi, v=v, v_dot=v_dot, a=a)

    def predict_q(self, q_n: StructureMinimalStates) -> StructureMinimalStates:
        r"""
        Predict the current state based upon the previous state.
        :param q_n: State at timestep n
        :return: Predicted state at timestep n+1
        """
        a = self.predict_a(v_dot_n=q_n.v_dot, a_n=q_n.a)
        phi = self.predict_phi(v_n=q_n.v, a_n=q_n.a, a_np1_pred=a)
        varphi = self.predict_varphi(varphi_n=q_n.varphi, phi_np1_pred=phi)
        v = self.predict_v(v_n=q_n.v, a_n=q_n.a, a_np1_pred=a)
        v_dot = self.predict_v_dot(v_dot_n=q_n.v_dot, a_n=q_n.a, a_np1_pred=a)
        return StructureMinimalStates(phi=phi, varphi=varphi, v=v, v_dot=v_dot, a=a)

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

    def calculate_varphi_alpha(self, varphi_nm1: Array, phi_n: Array) -> Array:
        phi_alpha = self.calculate_phi_alpha(phi_n)
        return vmap(
            lambda varphi_, phi_: log_se3(exp_se3(varphi_) @ exp_se3(phi_)), (0, 0), 0
        )(varphi_nm1, phi_alpha)

    def calculate_q_alpha(
        self, q_nm1: StructureMinimalStates, q_n: StructureMinimalStates
    ) -> StructureMinimalStates:
        phi_alpha = self.calculate_phi_alpha(phi_n=q_n.phi)
        varphi_alpha = self.calculate_varphi_alpha(
            varphi_nm1=q_nm1.varphi, phi_n=q_n.phi
        )
        v_alpha = self.calculate_v_alpha(v_nm1=q_nm1.v, v_n=q_n.v)
        v_dot_alpha = self.calculate_v_dot_alpha(
            v_dot_nm1=q_nm1.v_dot, v_dot_n=q_n.v_dot
        )
        a_alpha = self.calculate_a_alpha(a_nm1=q_nm1.a, a_n=q_n.a)
        return StructureMinimalStates(
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
