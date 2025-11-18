from jax import numpy as jnp
from jax import Array, jacobian
from jax.lax import cond
from aegrad.algebra.base import matrix2
from aegrad.algebra.so3 import vec_to_skew, skew_to_vec, t_so3, t_inv_so3, log_so3, exp_so3, bracket_neg_so3, alpha, beta
from aegrad.algebra.constants import SMALL_ANG_THRESH



def bracket_se3(vec1: Array, vec2: Array) -> Array:
    mat1 = ha_to_ha_tilde(vec1)
    mat2 = ha_to_ha_tilde(vec2)

    return mat1 @ mat2 - mat2 @ mat1

def bracket_neg_se3(vec1: Array, vec2: Array) -> Array:
    mat1 = ha_to_ha_tilde(vec1)
    mat2 = ha_to_ha_tilde(vec2)

    return mat1 @ mat2 + mat2 @ mat1


def t_u_omega_plus(ha: Array) -> Array:
    # [6] -> [3, 3]
    a = ha[:3]
    b = ha[3:]
    b_norm2 = jnp.inner(b, b)

    def t_u_omega_plus_full() -> Array:
        alpha_ = alpha(b)
        beta_ = beta(b)

        return (
            -0.5 * beta_ * vec_to_skew(a)
            + (1.0 - alpha_) / b_norm2 * bracket_neg_so3(a, b)
            + jnp.inner(b, a)
            / b_norm2
            * (
                (beta_ - alpha_) * vec_to_skew(b)
                + (0.5 * beta_ - 3.0 * (1.0 - alpha_) / b_norm2)
                * matrix2(vec_to_skew(b))
            )
        )

    def t_u_omega_plus_small_angle() -> Array:
        return -0.5 * vec_to_skew(a)

    return cond(b_norm2 > SMALL_ANG_THRESH, t_u_omega_plus_full, t_u_omega_plus_small_angle)


def t_u_omega_minus(ha: Array) -> Array:
    # [6] -> [3, 3]

    t_inv_ = t_inv_so3(ha[3:])
    return -t_inv_ @ t_u_omega_plus(ha) @ t_inv_

def t_se3(ha: Array) -> Array:
    # [6] -> [6, 6]
    t_ = t_so3(ha[3:])
    return jnp.block([[t_, t_u_omega_plus(ha)],
                      [jnp.zeros((3, 3)), t_]])

def t_inv_se3(ha: Array) -> Array:
    # [6] -> [6, 6]
    t_ = t_inv_so3(ha[3:])
    return jnp.block([[t_, t_u_omega_minus(ha)],
                      [jnp.zeros((3, 3)), t_]])


def log_se3(hg: Array) -> Array:
    # [4, 4] -> [6]
    omega = log_so3(hg[:3, :3])
    return jnp.concatenate((t_inv_so3(omega).T @ hg[:3, 3], omega))

def exp_se3(ha: Array) -> Array:
    # [6] -> [4, 4]
    return jnp.block([[exp_so3(ha[3:]), (t_so3(ha[3:]).T @ ha[:3])[:, None]],
                      [jnp.zeros((1, 3)), jnp.ones((1, 1))]])

def x_rmat_to_hg(x: Array, rmat: Array) -> Array:
    # [3], [3, 3] -> [4, 4]
    return jnp.block([[rmat, x[:, None]], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])

def hg_to_x_rmat(hg: Array) -> tuple[Array, Array]:
    # [4, 4] -> [3], [3, 3]
    return hg[:3, 3], hg[:3, :3]

def hg_inv(hg: Array) -> Array:
    # [4, 4] -> [4, 4]
    x, rmat = hg_to_x_rmat(hg)
    return jnp.block([[rmat.T, -(rmat.T @ x)[:, None]], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])

def ha_to_ha_tilde(ha: Array) -> Array:
    # [6] -> [4, 4]
    return jnp.block([[vec_to_skew(ha[3:]), ha[:3, None]], [jnp.zeros((1, 4))]])

def ha_tilde_to_ha(ha_tilde: Array) -> Array:
    # [4, 4] -> 6
    ha_u = ha_tilde[:3, 3]
    ha_omega = skew_to_vec(ha_tilde[:3, :3])
    return jnp.concatenate((ha_u, ha_omega), axis=-1)

def ha_to_ha_hat(ha: Array) -> Array:
    # [6] -> [6, 6]
    return jnp.block([[vec_to_skew(ha[3:]), vec_to_skew(ha[:3])], [jnp.zeros((3, 3)), vec_to_skew(ha[3:])]])

def ha_hat_to_ha(ha: Array) -> Array:
    # [6, 6] -> [6]
    ha_u = skew_to_vec(ha[:3, 3:])
    ha_omega = skew_to_vec(ha[:3, :3])
    return jnp.concatenate((ha_u, ha_omega), axis=-1)

def ha_to_ha_check(ha: Array) -> Array:
    # [6] -> [6, 6]
    return jnp.block([[jnp.zeros((3, 3)), vec_to_skew(ha[:3])], [vec_to_skew(ha[:3]), vec_to_skew(ha[3:])]])

def ha_check_to_ha(ha: Array) -> Array:
    # [6, 6] -> [6]
    ha_u = skew_to_vec(ha[:3, 3:])
    ha_omega = skew_to_vec(ha[3:, 3:])
    return jnp.concatenate((ha_u, ha_omega), axis=-1)

def hg_to_d(hg1: Array, hg2: Array) -> Array:
    # [4, 4], [4, 4] -> [6]
    return log_se3(hg_inv(hg1) @ hg2)

def p(d: Array) -> Array:
    # [6] -> [6, 12]
    return jnp.concatenate((-t_inv_se3(-d), t_inv_se3(d)), axis=-1)

def t_star(s_l: Array, d: Array) -> Array:
    # [], [6] -> [6, 6]
    return s_l * t_se3(s_l * d) @ t_inv_se3(d)

def q(s_l: Array, d: Array) -> Array:
    # [6] -> [6, 12]
    t_star_ = t_star(s_l, d)
    return jnp.stack((jnp.eye(6) - t_star_, t_star_), axis=-1)


def k_tg_entry(d: Array, eps: Array, k: Array) -> Array:
    def e_mat() -> Array:
        # [6] -> [12, 6, 12]

        d_pdt_dd = jacobian(lambda d_: p(d_).T)(d)  # [12, 6, 6]
        dd_dhab = p(d)  # [6, 12]
        return jnp.einsum("ijk,kl->ijl", d_pdt_dd, dd_dhab)  # [12, 6, 12]

    return jnp.einsum('ijk,jl,l->ik', e_mat(), k, eps)

def vect_product(hg: Array, x: Array) -> Array:
    # [4, 4], [3] -> [3]
    return hg[:3, :3] @ x + hg[:3, 3]
