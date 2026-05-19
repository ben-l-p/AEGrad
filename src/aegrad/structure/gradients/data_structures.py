from __future__ import annotations

from dataclasses import dataclass
import os
from math import prod
from pathlib import Path
from typing import Optional, Sequence, Self, TYPE_CHECKING, OrderedDict

import jax
from jax import Array, numpy as jnp

from aegrad.plotting.beam import plot_beam_to_vtk
from aegrad.algebra.array_utils import ArrayListShape
from aegrad.utils.utils import make_pytree
from aegrad.utils.data_structures import DesignVariables

if TYPE_CHECKING:
    from aegrad.structure.data_structures import (
        StaticStructure,
        DynamicStructureSnapshot,
    )


@jax.tree_util.register_dataclass
@dataclass
class StructureFullStates:
    v: Optional[Array]
    v_dot: Optional[Array]
    hg: Array
    eps: Array
    f_elem: Array
    f_res: Array  # residual forcing at nodes, added as it is used for trim


@dataclass(frozen=True)
class StructuralGradsToCompute:
    x0: bool = False
    orientation_euler: bool = False
    k_cs: bool = True
    m_cs: bool = True
    m_lumped: bool = False
    f_ext_follower: bool = False
    f_ext_dead: bool = False
    thrust_t: bool = False


@make_pytree
class StructuralDesignVariables(DesignVariables):
    def __init__(
        self,
        x0: Optional[Array],
        orientation_euler: Optional[Array],
        k_cs: Optional[Array],
        m_cs: Optional[Array],
        m_lumped: Optional[Array],
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        thrust_t: Optional[dict[str, Array]],
        f_shape: tuple[int, ...],
    ):
        super().__init__()
        self.x0: Optional[Array] = x0
        self.orientation_euler: Optional[Array] = orientation_euler
        self.k_cs: Optional[Array] = k_cs
        self.m_cs: Optional[Array] = m_cs
        self.m_lumped: Optional[Array] = m_lumped
        self.f_ext_follower: Optional[Array] = f_ext_follower
        self.f_ext_dead: Optional[Array] = f_ext_dead
        self.thrust_t: Optional[dict[str, Array]] = thrust_t
        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = prod(f_shape)

        self.shapes: OrderedDict[
            str,
            Optional[
                tuple[int, ...]
                | ArrayListShape
                | OrderedDict[str, tuple[int, ...] | ArrayListShape]
            ],
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def __iadd__(self, other: StructuralDesignVariables) -> Self:
        if self.x0 is not None:
            assert other.x0 is not None
            self.x0 += other.x0
        if self.orientation_euler is not None:
            assert other.orientation_euler is not None
            self.orientation_euler += other.orientation_euler
        if self.k_cs is not None:
            assert other.k_cs is not None
            self.k_cs += other.k_cs
        if self.m_cs is not None:
            assert other.m_cs is not None
            self.m_cs += other.m_cs
        if self.m_lumped is not None:
            assert other.m_lumped is not None
            self.m_lumped += other.m_lumped
        if self.f_ext_follower is not None:
            assert other.f_ext_follower is not None
            self.f_ext_follower += other.f_ext_follower
        if self.f_ext_dead is not None:
            assert other.f_ext_dead is not None
            self.f_ext_dead += other.f_ext_dead
        if self.thrust_t is not None:
            assert other.thrust_t is not None
            for k in self.thrust_t.keys():
                self.thrust_t[k] += other.thrust_t[k]
        return self

    def premultiply_adj(self, adj: Array) -> StructuralDesignVariables:
        return StructuralDesignVariables(
            x0=jnp.einsum("ij,j...->i...", adj, self.x0)
            if self.x0 is not None
            else None,
            orientation_euler=jnp.einsum("ij,j...->i...", adj, self.orientation_euler)
            if self.orientation_euler is not None
            else None,
            k_cs=jnp.einsum("ij,j...->i...", adj, self.k_cs)
            if self.k_cs is not None
            else None,
            m_cs=jnp.einsum("ij,j...->i...", adj, self.m_cs)
            if self.m_cs is not None
            else None,
            m_lumped=jnp.einsum("ij,j...->i...", adj, self.m_lumped)
            if self.m_lumped is not None
            else None,
            f_ext_follower=jnp.einsum("ij,j...->i...", adj, self.f_ext_follower)
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=jnp.einsum("ij,j...->i...", adj, self.f_ext_dead)
            if self.f_ext_dead is not None
            else None,
            thrust_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.thrust_t.items()
            }
            if self.thrust_t is not None
            else None,
            f_shape=(adj.shape[1],),
        )

    def zeros_like(self) -> StructuralDesignVariables:
        return StructuralDesignVariables(
            x0=jnp.zeros_like(self.x0) if self.x0 is not None else None,
            orientation_euler=jnp.zeros_like(self.orientation_euler)
            if self.orientation_euler is not None
            else None,
            k_cs=jnp.zeros_like(self.k_cs) if self.k_cs is not None else None,
            m_cs=jnp.zeros_like(self.m_cs) if self.m_cs is not None else None,
            m_lumped=jnp.zeros_like(self.m_lumped)
            if self.m_lumped is not None
            else None,
            f_ext_follower=jnp.zeros_like(self.f_ext_follower)
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=jnp.zeros_like(self.f_ext_dead)
            if self.f_ext_dead is not None
            else None,
            thrust_t={k: jnp.zeros_like(v) for k, v in self.thrust_t.items()}
            if self.thrust_t is not None
            else None,
            f_shape=self.f_shape,
        )

    def plot(
        self,
        case: StaticStructure | DynamicStructureSnapshot,
        directory: os.PathLike | str,
        n_interp: int = 0,
    ) -> Path:

        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        d_f_d_f_follower = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_follower[:, :3])
            if self.f_ext_follower is not None
            else None
        )
        d_f_d_m_follower = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_follower[:, 3:])
            if self.f_ext_follower is not None
            else None
        )
        d_f_d_f_dead = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_dead[:, :3])
            if self.f_ext_dead is not None
            else None
        )
        d_f_d_m_dead = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_dead[:, 3:])
            if self.f_ext_dead is not None
            else None
        )
        d_f_d_x0 = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.x0)
            if self.x0 is not None
            else None
        )

        node_num = jnp.arange(case.hg.shape[0])  # [n_nodes]
        elem_num = jnp.arange(case.conn.shape[0])  # [n_elems]

        node_scalar_data = {"node_number": node_num}
        node_vector_data = {
            "d_f_d_f_follower": d_f_d_f_follower,
            "d_f_d_m_follower": d_f_d_m_follower,
            "d_f_d_f_dead": d_f_d_f_dead,
            "d_f_d_m_dead": d_f_d_m_dead,
            "d_f_d_x0": d_f_d_x0,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {}

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(directory).joinpath(
            "beam_gradient"
        )  # default file name for beam objects is "beam"
        return plot_beam_to_vtk(
            hg=case.hg,
            conn=case.conn,
            o0=case.o0,
            n_interp=n_interp,
            filename=file_name,
            i_ts=None,
            node_scalar_data=node_scalar_data,
            node_vector_data=node_vector_data,
            cell_scalar_data=cell_scalar_data,
            cell_vector_data=cell_vector_data,
        )

    def get_vars(self) -> dict[str, Optional[Array] | Optional[dict[str, Array]]]:
        return {
            "x0": self.x0,
            "orientation_euler": self.orientation_euler,
            "k_cs": self.k_cs,
            "m_cs": self.m_cs,
            "m_lumped": self.m_lumped,
            "f_ext_follower": self.f_ext_follower,
            "f_ext_dead": self.f_ext_dead,
            "thrust_t": self.thrust_t,
        }

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "x0",
            "orientation_euler",
            "k_cs",
            "m_cs",
            "m_lumped",
            "f_ext_follower",
            "f_ext_dead",
            "thrust_t",
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "f_shape", "f_size"
