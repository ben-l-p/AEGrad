from jax import Array, numpy as jnp, jacobian

from algebra.se3 import p


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


def g_int_entry(
    p_d: Array,
    k: Array,
    eps: Array,
) -> Array:
    r"""
    Computes the nodal internal force vector for a beam element. Formulation from Geometrically exact beam
    finite element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 75.
    :param p_d: :math:`\mathbf{P}(\mathbf{d})` matrix, [6, 12].
    :param k: Beam cross-sectional stiffness matrix, [6, 6].
    :param eps: Beam strain vector, [6].
    :return: Forces at nodes, [12].
    """
    return p_d.T @ k @ eps


def k_t_entry(
    d: Array,
    p_d: Array,
    l: Array,
    eps: Array,
    k: Array,
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
        # contribution from perturbations in P(d)
        k_t += jacobian(lambda d_: p(d_).T @ k @ eps)(d) @ p_d
    return k_t
