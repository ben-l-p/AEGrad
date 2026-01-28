from __future__ import annotations
from copy import deepcopy
from os import PathLike
from pathlib import Path

from jax import numpy as jnp
from jax import Array
from typing import Optional
from aegrad.utils import make_pytree
from typing import Sequence
from aegrad.print_output import warn
from aegrad.algebra.base import chi
from jax import vmap

from plotting.beam import plot_beam_to_vtk


class StaticStructure:
    """Results of a static structure analysis."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        d: Array,
        eps: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_int: Array,
        local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.local: bool = local

    def to_dynamic(self) -> DynamicStructureSnapshot:
        """Convert static structure results to a dynamic structure snapshot with zero velocities."""
        n_nodes = self.hg.shape[0]
        n_elem = self.d.shape[0]
        zero_d_dot = jnp.zeros((n_elem, 6))
        zero_v = jnp.zeros((n_nodes, 6))
        zero_v_dot = jnp.zeros((n_nodes, 6))
        zero_time = jnp.array(0.0)

        return DynamicStructureSnapshot(
            hg=self.hg,
            conn=self.conn,
            d=self.d,
            d_dot=zero_d_dot,
            eps=self.eps,
            v=zero_v,
            v_dot=zero_v_dot,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_int=self.f_int,
            t=zero_time,
            i_ts=-1,
        )

    def _transform(self, nodal_chi: Array) -> None:
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_follower)
            )
        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_dead)
            )
        self.f_int = self.f_int.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_int)
        )

    def to_global(self) -> None:
        """Convert local structure results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def plot(self, directory: PathLike) -> Path:
        r"""
        Plot beam results to VTK files in the specified directory.
        """
        return self.to_dynamic().plot(directory)


class DynamicStructureSnapshot:
    """Snapshot of dynamic structure results at a specific time."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        d: Array,
        d_dot: Array,
        eps: Array,
        v: Array,
        v_dot: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_int: Array,
        t: Array,
        i_ts: int,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_elem, 6]
        self.d_dot: Array = d_dot  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.v: Array = v  # [n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.t: Array = t  # Scalar time value
        self.i_ts: int = i_ts  # Time step index

    def to_static(self) -> StaticStructure:
        """Extract static structure results from the snapshot."""
        return StaticStructure(
            hg=self.hg,
            conn=self.conn,
            d=self.d,
            eps=self.eps,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_int=self.f_int,
        )

    def _transform(self, nodal_chi: Array) -> None:
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_follower)
            )
        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_dead)
            )
        self.f_int = self.f_int.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_int)
        )
        self.v = self.v.at[...].set(jnp.einsum("ijk,ik->ij", nodal_chi, self.v))
        self.v_dot = self.v_dot.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.v_dot)
        )

    def to_global(self) -> None:
        """Convert local structure results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def plot(self, directory: PathLike) -> Path:
        data = deepcopy(self)
        data.to_global()

        coords = data.hg[:, :3, 3]  # [n_nodes, 3]

        local_x = data.hg[:, :3, 0]  # [n_nodes, 3]
        local_y = data.hg[:, :3, 1]  # [n_nodes, 3]
        local_z = data.hg[:, :3, 2]  # [n_nodes, 3]

        f_ext_follower = (
            data.f_ext_follower[:, :3] if data.f_ext_follower is not None else None
        )  # [n_nodes, 3]
        m_ext_follower = (
            data.f_ext_follower[:, 3:] if data.f_ext_follower is not None else None
        )  # [n_nodes, 3]
        f_ext_dead = (
            data.f_ext_dead[:, :3] if data.f_ext_dead is not None else None
        )  # [n_nodes, 3]
        m_ext_dead = (
            data.f_ext_dead[:, 3:] if data.f_ext_dead is not None else None
        )  # [n_nodes, 3]
        f_int = data.f_int[:, :3]  # [n_nodes, 3]
        m_int = data.f_int[:, 3:]  # [n_nodes, 3]

        v_lin = data.v[:, :3]  # [n_nodes, 3]
        v_ang = data.v[:, 3:]  # [n_nodes, 3]
        eps_lin = data.eps[:, :3]  # [n_elem, 3]
        eps_ang = data.eps[:, 3:]  # [n_elem, 3]

        node_num = jnp.arange(data.hg.shape[0])  # [n_nodes]
        elem_num = jnp.arange(data.conn.shape[0])  # [n_elems]

        node_scalar_data = {"node_number": node_num}
        node_vector_data = {
            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,
            "f_ext_follower": f_ext_follower,
            "m_ext_follower": m_ext_follower,
            "f_ext_dead": f_ext_dead,
            "m_ext_dead": m_ext_dead,
            "f_int": f_int,
            "m_int": m_int,
            "v_linear": v_lin,
            "v_angular": v_ang,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {"eps_linear": eps_lin, "eps_angular": eps_ang}

        return plot_beam_to_vtk(
            coords,
            data.conn,
            directory,
            data.i_ts,
            node_scalar_data,
            node_vector_data,
            cell_scalar_data,
            cell_vector_data,
        )


@make_pytree
class DynamicStructure:
    """Results of a dynamic structure analysis."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        d: Array,
        d_dot: Array,
        eps: Array,
        v: Array,
        v_dot: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_int: Array,
        t: Array,
    ):
        self.hg: Array = hg  # [n_tstep, n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_tstep, n_elem, 6]
        self.d_dot: Array = d_dot  # [n_tstep, n_elem, 6]
        self.eps: Array = eps  # [n_tstep, n_elem, 6]
        self.v: Array = v  # [n_tstep, n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_tstep, n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_tstep, n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_tstep, n_nodes, 6]
        self.f_int: Array = f_int  # [n_tstep, n_nodes, 6]
        self.t: Array = t  # [n_tstep]

    def to_static(self, i_ts: int) -> StaticStructure:
        """Extract static structure results at a specific time index."""
        return StaticStructure(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            f_ext_follower=self.f_ext_follower[i_ts, ...]
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=self.f_ext_dead[i_ts, ...]
            if self.f_ext_dead is not None
            else None,
            f_int=self.f_int[i_ts, ...],
        )

    def __getitem__(self, i_ts: int) -> DynamicStructureSnapshot:
        """Extract dynamic structure snapshot at a specific time index."""
        return DynamicStructureSnapshot(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            d=self.d[i_ts, ...],
            d_dot=self.d_dot[i_ts, ...],
            eps=self.eps[i_ts, ...],
            v=self.v[i_ts, ...],
            v_dot=self.v_dot[i_ts, ...],
            f_ext_follower=self.f_ext_follower[i_ts, ...]
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=self.f_ext_dead[i_ts, ...]
            if self.f_ext_dead is not None
            else None,
            f_int=self.f_int[i_ts, ...],
            t=self.t[i_ts],
            i_ts=i_ts,
        )

    @classmethod
    def initialise(
        cls, initial_snapshot: DynamicStructureSnapshot, n_tstep: int
    ) -> DynamicStructure:
        r"""
        Initialise a DynamicStructure object given an initial snapshot and number of time steps.
        :param initial_snapshot: Snapshot at initial time step.
        :param n_tstep: Number of time steps in the dynamic analysis. This includes the initial time step at t=0.
        :return: DynamicStructure object with arrays initialised to zero except for the first time step.
        """
        n_node = initial_snapshot.hg.shape[0]
        n_elem = initial_snapshot.d.shape[0]

        hg = jnp.zeros((n_tstep, n_node, 4, 4)).at[0, ...].set(initial_snapshot.hg)
        conn = initial_snapshot.conn
        d = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d)
        d_dot = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d_dot)
        eps = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.eps)
        v = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v)
        v_dot = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v_dot)
        f_ext_follower = (
            jnp.zeros((n_tstep, n_node, 6))
            .at[0, ...]
            .set(initial_snapshot.f_ext_follower)
        )
        f_ext_dead = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_ext_dead)
        )
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        t = jnp.zeros((n_tstep,)).at[0].set(initial_snapshot.t)
        return cls(
            hg, conn, d, d_dot, eps, v, v_dot, f_ext_follower, f_ext_dead, f_int, t
        )

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Get names of static attributes in dynamic beam
        """
        return ("conn",)

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Get names of dynamic attributes in dynamic beam
        """
        return (
            "hg",
            "d",
            "d_dot",
            "eps",
            "v",
            "v_dot",
            "f_ext_follower",
            "f_ext_dead",
            "f_int",
            "t",
        )
