from jax import numpy as jnp
from aegrad.structure.structure import Structure
from typing import Literal, Optional
from jax import Array
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)


def geradin_beam(
    n_nodes: int = 20,
    beam_direction: Literal["x", "y", "z"] = "x",
    m_cs: Optional[Array] = None,
) -> Structure:
    l = jnp.array(5.0)
    n_elem = n_nodes - 1

    direction_index = {"x": 0, "y": 1, "z": 2}[beam_direction]
    y_vect = {
        "x": jnp.array([0.0, 1.0, 0.0]),
        "y": jnp.array([0.0, 0.0, 1.0]),
        "z": jnp.array([1.0, 0.0, 0.0]),
    }[beam_direction]

    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    coords = (
        jnp.zeros((n_nodes, 3)).at[:, direction_index].set(jnp.linspace(0, l, n_nodes))
    )
    struct = Structure(n_nodes, conn, y_vect[None, :])

    k_coeffs = jnp.full(6, 1e12)
    k_coeffs = k_coeffs.at[1:3].set(3.231e8)
    k_coeffs = k_coeffs.at[4:6].set(9.345e6)
    struct.set_design_variables(coords, jnp.diag(k_coeffs), m_cs)
    return struct


if __name__ == "__main__":
    n_nodes = 20
    load = 600000.0
    struct = geradin_beam(n_nodes=n_nodes, beam_direction="x")
    f_ext = jnp.zeros((n_nodes, 6))
    f_ext = f_ext.at[-1, 2].set(-load)

    result = struct.static_solve(
        None,
        f_ext,
        jnp.arange(6),
        max_iter=30,
    )

    out_path = Path("./output/geradin_beam")
    result.plot(out_path, 3)
