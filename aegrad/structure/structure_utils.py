from jax import Array, numpy as jnp
import jax
from aegrad.algebra.se3 import p, q, q_dot, ha_to_ha_hat, ha_to_ha_check
from aegrad.algebra.integration import gauss_lobatto, gauss_legendre
from typing import Literal


def check_connectivity(connectivity: Array, num_nodes: int) -> None:
    r"""
    Check connectivity array for validity
    :param connectivity: Connectivity array of shapes [n_elem, 2]
    :param num_nodes: Number of nodes in the structure
    :raises ValueError: If connectivity array contains invalid node indices, or is missing nodes
    """
    all_node_index = set(connectivity.ravel().tolist())
    expected_node_index = set(range(num_nodes))

    if all_node_index != expected_node_index:
        raise ValueError(
            f"Connectivity array either contains invalid node indices, or is missing nodes. Expected "
            f"indices: {expected_node_index}, but got: {all_node_index}."
        )


def n_elem_per_node(connectivity: Array) -> Array:
    r"""
    Computes the number of elements connected to each node in the structure.
    :param connectivity: Connectivity array of shape [n_elem, 2]
    :return: Number of elements connected to each node, [num_nodes]
    """

    return jnp.bincount(
        connectivity.ravel(),
        minlength=connectivity.shape[0],
        length=connectivity.shape[0],
    )


def k_t_entry(
    d: Array,
    p_d: Array,
    l: Array,
    eps: Array,
    k: Array,
    ad_inv: Array,
    include_material: bool = True,
    include_geometric: bool = True,
) -> Array:
    r"""
    Computes a stiffness matrix entry between two degrees of freedom. Formulation from Geometrically exact beam finite
    element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 76.
    :param d: Relative se(3) configuration vector, [6].
    :param p_d: :math:`\mathbf{P}(\mathbf{d})` matrix, [6, 12].
    :param l: Element length, [].
    :param eps: Beam strain vector, [6].
    :param k: Beam cross-sectional stiffness matrix, [6, 6].
    :param ad_inv: Inverse adjoint matrix for element, [6, 6].
    :param include_material: Whether to include material stiffness contribution, bool.
    :param include_geometric: Whether to include geometric stiffness contribution, bool.
    :return: Stiffness matrix entry, [12, 12].
    """
    if not include_material and not include_geometric:
        raise ValueError(
            "At least one of include_material or include_geometric must be True."
        )

    if include_material:
        # contribution from perturbations in the strain
        k_t = p_d.T @ k @ p_d / l  # [12, 12]
    else:
        k_t = jnp.zeros((12, 12))

    if include_geometric:
        e = jax.jacobian(lambda d__,: p(d__, ad_inv))(d)

        # [12, 6, 12]
        f = jnp.einsum("ijk, kl->ijl", e, p_d)

        # [12, 12]
        k_tg = jnp.einsum("jil,j->il", f, (k * l) @ eps)
        k_t += k_tg
    return k_t


def integrate_m_l(
    m_cs: Array,
    d: Array,
    ad_inv: Array,
    l: Array,
    int_order: Literal[3, 4, 5],
) -> Array:
    r"""
    Approximate the integral :math:`\int_L \mathbf{Q}(s, \mathbf{d})^{\top} \mathcal{M}_{CS} \mathbf{Q}(s, \mathbf{d}) \ ds`
    :param m_cs: Cross sectional mass matrix, [6, 6].
    :param d: Configuration vector, [6]
    :param ad_inv: Inverse adjoint matrix for element, [6, 6]
    :param l: Element length, []
    :param int_order: Order of integration, 3, 4, or 5.
    :return: Integrated mass matrix, [12, 12]
    """

    def inner_func(s_l) -> Array:
        q_mat = q(s_l, d, ad_inv)
        return q_mat.T @ m_cs @ q_mat

    f0 = jnp.zeros((12, 12)).at[:6, :6].set(ad_inv.T @ m_cs @ ad_inv)
    fl = jnp.zeros((12, 12)).at[6:, 6:].set(ad_inv.T @ m_cs @ ad_inv)

    return l * gauss_lobatto(
        inner_func,
        jnp.array((0.0, 1.0)),
        jnp.stack((f0, fl), axis=0),
        int_order=int_order,
    )


def integrate_c_t(
    m_cs: Array,
    v_ab: Array,
    d: Array,
    d_dot: Array,
    ad_inv: Array,
    l: Array,
    int_order: Literal[1, 2, 3],
) -> Array:
    r"""
    Approximate the integral :math:`C_T = C^L - \int_L \check{(\mathbf{MQv}_{AB})^{\top} \mathbf{Q} \ ds` where
    :math:`C^L = \int_L \mathbf{Q}^{\top} ( \mathbf{M}_{cs} \dot{\mathbf{Q}} - \hat{\mathbf{Qv}_{AB}}^{\top}
    \mathbf{M}_{cs} \mathbf{Q} ) \ ds`
    :param m_cs: Cross sectional mass matrix, [6, 6].
    :param v_ab: Nodal local velocities, [12]
    :param d: Configuration vector, [6]
    :param d_dot: Configuration velocity vector, [6]
    :param ad_inv: Inverse adjoint matrix for element, [6, 6]
    :param l: Element length, []
    :param int_order: Order of integration, 1, 2, or 3
    :return: Stacked [C_L, C_T], [2, 12, 12]
    """

    def inner_func(s_l) -> Array:
        q_mat = q(s_l, d, ad_inv)
        q_dot_mat = q_dot(s_l, d, d_dot, ad_inv)
        c_l = q_mat.T @ (m_cs @ q_dot_mat - ha_to_ha_hat(q_mat @ v_ab).T @ m_cs @ q_mat)
        c_t = c_l - ha_to_ha_check(m_cs @ q_mat @ v_ab).T @ q_mat
        return jnp.stack((c_l, c_t), axis=0)

    return l * gauss_legendre(
        inner_func,
        jnp.array((0.0, 1.0)),
        int_order=int_order,
    )  # [2, 12, 12]
