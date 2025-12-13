from jax import Array


def check_connectivity(connectivity: Array, num_nodes: int) -> None:
    r"""
    Check connectivity array for validity
    :param connectivity: Connectivity array of shapes [n_elem, 2]
    :param num_nodes: Number of nodes in the structure
    """
    all_node_index = set(connectivity.ravel().tolist())
    expected_node_index = set(range(num_nodes))

    if all_node_index != expected_node_index:
        raise ValueError(
            f"Connectivity array either contains invalid node indices, or is missing nodes. Expected "
            f"indices: {expected_node_index}, but got: {all_node_index}."
        )
