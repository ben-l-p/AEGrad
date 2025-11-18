from jax import numpy as jnp
from jax import Array
from jax.lax import cond
from jax.scipy.special import bernoulli
from math import factorial
from aegrad.algebra.base import clip_to_pi, matrix2
from aegrad.algebra.constants import ZERO_ANG_THRESH, SMALL_ANG_THRESH

def vec_to_skew(vec: Array) -> Array:
    # [3] -> [3, 3]
    return jnp.array(((0.0, -vec[2], vec[1]),
                     (vec[2], 0.0, -vec[0]),
                     (-vec[1], vec[0], 0.0)))

def skew_to_vec(mat: Array) -> Array:
    # [3, 3] -> [3]
    return jnp.array((mat[2, 1], mat[0, 2], mat[1, 0]))

def alpha(b: Array) -> Array:
    # [3] -> []
    b_norm = jnp.linalg.norm(b)

    def alpha_full() -> Array:
        return jnp.sin(b_norm) / b_norm

    return cond(b_norm > ZERO_ANG_THRESH, alpha_full, lambda: 1.0)


def beta(b: Array) -> Array:
    # [3] -> []
    b_norm2 = jnp.inner(b, b)
    b_norm = jnp.sqrt(b_norm2)

    def beta_full() -> Array:
        return 2.0 * (1.0 - jnp.cos(b_norm)) / b_norm2

    return cond(b_norm > ZERO_ANG_THRESH, beta_full, lambda: 1.0)

def bound_h_omega(h_omega: Array) -> Array:
    # [3] -> [3]
    # sets the angle for a CRV to be within pi limits
    ang = jnp.linalg.norm(h_omega)

    def nonzero_ang() -> Array:
        n = h_omega / ang
        bounded_ang = clip_to_pi(ang)
        return bounded_ang * n

    def zero_ang() -> Array:
        return h_omega

    return cond(ang > ZERO_ANG_THRESH, nonzero_ang, zero_ang)


def bracket_so3(vec1: Array, vec2: Array) -> Array:
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 - mat2 @ mat1

def bracket_neg_so3(vec1: Array, vec2: Array) -> Array:
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 + mat2 @ mat1


def t_so3(ha_omega: Array) -> Array:
    # [3] -> [3, 3]
    def t_so3_full() -> Array:
        return (jnp.eye(3) - 0.5 * beta(ha_omega) * vec_to_skew(ha_omega)
                + (1.0 - alpha(ha_omega)) / jnp.inner(ha_omega, ha_omega) * matrix2(vec_to_skew(ha_omega)))

    def t_so3_small_angle() -> Array:
        order: int = 2
        skew = vec_to_skew(ha_omega)
        out = jnp.eye(3)
        for i in range(1, order + 1):
            out += (-1.0) ** i / factorial(i + 1) * jnp.linalg.matrix_power(skew, i)
        return out

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_so3_full, t_so3_small_angle)

def t_inv_so3(ha_omega: Array) -> Array:
    # [3] -> [3, 3]

    def t_inv_so3_full() -> Array:
        return (jnp.eye(3) + 0.5 * vec_to_skew(ha_omega)
            + (1.0 - alpha(ha_omega) / beta(ha_omega)) / jnp.inner(ha_omega, ha_omega) * matrix2(vec_to_skew(ha_omega)))

    def t_inv_so3_small_angle() -> Array:
        order: int = 2
        skew = vec_to_skew(ha_omega)
        b = bernoulli(order + 1)
        out = jnp.eye(3)
        for i in range(1, order + 1):
            out += (-1.0) ** i * b[i] * jnp.linalg.matrix_power(skew, i) / factorial(i)
        return out

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_inv_so3_full, t_inv_so3_small_angle)


def exp_so3(ha_omega: Array) -> Array:
    # [3] -> [3, 3]
    def exp_so3_full() -> Array:
        # has a singularity as ha_omega -> 0
        ang = jnp.linalg.norm(ha_omega)
        return jnp.eye(3) + jnp.sin(ang) / ang * vec_to_skew(ha_omega) + (1.0 - jnp.cos(ang)) / ang ** 2 * matrix2(vec_to_skew(ha_omega))

    def exp_so3_small_angle() -> Array:
        return jnp.eye(3) + vec_to_skew(ha_omega) + 0.5 * matrix2(vec_to_skew(ha_omega))

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH,
                exp_so3_full,
                exp_so3_small_angle)

def log_so3(rmat: Array) -> Array:
    # [3, 3] -> [3]
    theta = jnp.acos(0.5 * (jnp.trace(rmat) - 1.0))

    def log_so3_full() -> Array:
        return skew_to_vec(theta / (2.0 * jnp.sin(theta)) * (rmat - rmat.T))

    def log_so3_small_angle() -> Array:
        order: int = 1
        out = jnp.zeros((3, 3))
        for i in range(1, order):
            out += (-1.0) ** (i + 1) * jnp.linalg.matrix_power((rmat - jnp.eye(3)), i) / i
        return skew_to_vec(out)

    return bound_h_omega(cond(theta > SMALL_ANG_THRESH, log_so3_full, log_so3_small_angle))