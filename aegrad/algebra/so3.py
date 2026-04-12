from jax import numpy as jnp
from jax import Array
from jax.lax import cond

from algebra.base import clip_to_pi, matrix2, t_sum, t_inv_sum, exp_sum, log_sum
from constants import SMALL_ANG_THRESH


def vec_to_skew(vec: Array) -> Array:
    r"""
    Converts a 3D vector to a skew-symmetric matrix.
    :param vec: 3D vector, [3]
    :return: Skew-symmetric matrix, [3, 3]
    """
    return jnp.array(
        ((0.0, -vec[2], vec[1]), (vec[2], 0.0, -vec[0]), (-vec[1], vec[0], 0.0))
    )


def skew_to_vec(mat: Array) -> Array:
    r"""
    Converts a skew-symmetric matrix to a 3D vector. Note this refers to both skew symmetric entries for consistent
    gradients.
    :param mat: Skew-symmetric matrix, [3, 3]
    :return: 3D vector, [3]
    """
    a1 = 0.5 * (mat[2, 1] - mat[1, 2])
    a2 = 0.5 * (mat[0, 2] - mat[2, 0])
    a3 = 0.5 * (mat[1, 0] - mat[0, 1])

    return jnp.array((a1, a2, a3))


def alpha(b: Array) -> Array:
    r"""
    Computes the alpha function for SO(3) operations. This includes a small angle approximation as b approaches zero.
    Formulation from Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by
    Sonneville et al., 2013, Eq A.4
    :param b: Input vector, [3]
    :return: Alpha value, []
    """
    b_norm = jnp.linalg.norm(b)

    def alpha_full() -> Array:
        return jnp.sin(b_norm) / b_norm

    def alpha_small_angle() -> Array:
        return 1.0 - b_norm ** 2 / 6.0

    return cond(b_norm > SMALL_ANG_THRESH, alpha_full, alpha_small_angle)


def beta(b: Array) -> Array:
    r"""
    Computes the beta function for SO(3) operations. This includes a small angle approximation as b approaches zero.
    Formulation from Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by
    Sonneville et al., 2013, Eq A.4
    :param b: Input vector, [3]
    :return: Beta value, []
    """
    b_norm2 = jnp.inner(b, b)
    b_norm = jnp.sqrt(b_norm2)

    def beta_full() -> Array:
        return 2.0 * (1.0 - jnp.cos(b_norm)) / b_norm2

    def beta_small_angle() -> Array:
        return 0.5 - b_norm2 / 24.0

    return cond(b_norm > SMALL_ANG_THRESH, beta_full, beta_small_angle)


def bound_h_omega(h_omega: Array) -> Array:
    r"""
    Bounds the angle of a rotation vector to be within [-pi, pi].
    :param h_omega: Cartesian rotation vector, [3]
    :return: Bounded Cartesian rotation vector, [3]
    """
    ang = jnp.linalg.norm(h_omega)

    def nonzero_ang() -> Array:
        n = h_omega / ang
        bounded_ang = clip_to_pi(ang)
        return bounded_ang * n

    def small_ang() -> Array:
        return h_omega

    return cond(ang > SMALL_ANG_THRESH, nonzero_ang, small_ang)


def bracket_so3(vec1: Array, vec2: Array) -> Array:
    r"""
    Computes the Lie bracket of two so(3) elements represented as vectors.
    :param vec1: Vector 1, [3]
    :param vec2: Vector 2, [3]
    :return: Lie bracket, [3, 3]
    """
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 - mat2 @ mat1


def bracket_neg_so3(vec1: Array, vec2: Array) -> Array:
    r"""
    Computes the negative Lie bracket of two so(3) elements represented as vectors.
    :param vec1: Vector 1, [3]
    :param vec2: Vector 2, [3]
    :return: Negative Lie bracket, [3, 3]
    """
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 + mat2 @ mat1


def t_so3(ha_omega: Array) -> Array:
    r"""
    Computes the tangent operator for SO(3) given a rotation vector. Includes a small angle approximation as the
    rotation approaches zero. Formulation from Geometrically exact beam finite element formulated on the special
    Euclidean group SE(3), by Sonneville et al., 2013, Eq A.6
    :param ha_omega: Rotation vector, [3]
    :return: Tangent operator, [3, 3]
    """

    def t_so3_full() -> Array:
        return (
                jnp.eye(3)
                - 0.5 * beta(ha_omega) * vec_to_skew(ha_omega)
                + (1.0 - alpha(ha_omega))
                / jnp.inner(ha_omega, ha_omega)
                * matrix2(vec_to_skew(ha_omega))
        )

    def t_so3_small_angle() -> Array:
        return t_sum(vec_to_skew(ha_omega), 2)

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_so3_full, t_so3_small_angle)


def t_inv_so3(ha_omega: Array) -> Array:
    r"""
    Computes the inverse tangent operator for SO(3) given a rotation vector. Includes a small angle approximation as the
    rotation approaches zero. Formulation from Geometrically exact beam finite element formulated on the special
    Euclidean group SE(3), by Sonneville et al., 2013, Eq A.7
    :param ha_omega: Rotation vector, [3]
    :return: Inverse tangent operator, [3, 3]
    """

    def t_inv_so3_full() -> Array:
        return (
                jnp.eye(3)
                + 0.5 * vec_to_skew(ha_omega)
                + (1.0 - alpha(ha_omega) / beta(ha_omega))
                / jnp.inner(ha_omega, ha_omega)
                * matrix2(vec_to_skew(ha_omega))
        )

    def t_inv_so3_small_angle() -> Array:
        return t_inv_sum(vec_to_skew(ha_omega), 2)

    ang_mag2 = jnp.linalg.norm(ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_inv_so3_full, t_inv_so3_small_angle)


def exp_so3(ha_omega: Array) -> Array:
    r"""
    Computes the exponential map from so(3) to SO(3) given a rotation vector. Includes a small angle approximation as
    the angle approaches zero.
    :param ha_omega: Rotation vector, [3]
    :return: Rotation matrix, [3, 3]
    """

    def exp_so3_full() -> Array:
        # has a singularity as ha_omega -> 0
        ang = jnp.linalg.norm(ha_omega)
        return (
                jnp.eye(3)
                + jnp.sin(ang) / ang * vec_to_skew(ha_omega)
                + (1.0 - jnp.cos(ang)) / ang ** 2 * matrix2(vec_to_skew(ha_omega))
        )

    def exp_so3_small_angle() -> Array:
        return exp_sum(vec_to_skew(ha_omega), 2)

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, exp_so3_full, exp_so3_small_angle)


def log_so3(rmat: Array) -> Array:
    r"""
    Computes the logarithmic map from SO(3) to so(3) given a rotation matrix. Includes a small angle approximation as
    the angle approaches zero.
    :param rmat: Rotation matrix, [3, 3]
    :return: Rotation vector, [3]
    """

    cos_theta = 0.5 * (jnp.trace(rmat) - 1.0)

    def log_so3_full() -> Array:
        # acos is only computed here, where theta is large enough that its gradient is finite
        theta = jnp.acos(cos_theta)
        return skew_to_vec(theta / (2.0 * jnp.sin(theta)) * (rmat - rmat.T))

    def log_so3_small_angle() -> Array:
        return skew_to_vec(log_sum(rmat, 2))

    # theta > SMALL_ANG_THRESH  ⟺  cos_theta < cos(SMALL_ANG_THRESH)
    return cond(cos_theta < jnp.cos(SMALL_ANG_THRESH), log_so3_full, log_so3_small_angle)
