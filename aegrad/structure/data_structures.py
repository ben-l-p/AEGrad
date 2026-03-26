from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass

from jax import numpy as jnp, Array
from jax import vmap

from aegrad.print_utils import warn
from aegrad.algebra.base import chi
from aegrad.plotting.beam import plot_beam_to_vtk
from aegrad.plotting.pvd import write_pvd
from aegrad.utils import _make_pytree


@dataclass
class OptionalJacobians:
    d_f_ext_dead_d_n: bool = False  # stiffness from dead loads
    d_f_grav_d_n: bool = False  # stiffness from gravitational loads
    d_f_gyr_d_q_dot: bool = (
        False  # derivative of gyroscopic forces with respect to Q_dot
    )
    d_f_int_d_p_d: bool = False  # geometric stiffness


@_make_pytree
class StaticStructure:
    """Object to hold the full state and forces of a static structure_dv analysis."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        o0: Array,
        d: Array,
        eps: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_ext_aero: Optional[Array],
        f_grav: Optional[Array],
        f_int: Array,
        f_res: Array,
        prescribed_dofs: Optional[Array],
        local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes_, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes_, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes_, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_nodes_, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes_, 6]
        self.f_int: Array = f_int  # [n_nodes_, 6]
        self.f_res: Array = f_res  # [n_nodes_, 6]
        self.prescribed_dofs: Optional[Array] = prescribed_dofs
        self.local: bool = local

    def to_dynamic(self) -> DynamicStructureSnapshot:
        """Convert static structure_dv results to a dynamic structure_dv snapshot, with zeroed velocity/acceleration dependent
        entries."""
        n_nodes = self.hg.shape[0]
        zero_v = jnp.zeros((n_nodes, 6))
        zero_v_dot = jnp.zeros((n_nodes, 6))
        zero_a = jnp.zeros((n_nodes, 6))
        zero_time = jnp.array(0.0)
        zero_f_iner = jnp.zeros((n_nodes, 6))

        return DynamicStructureSnapshot(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            v=zero_v,
            v_dot=zero_v_dot,
            a=zero_a,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_iner=zero_f_iner,
            f_res=self.f_res,
            t=zero_time,
            i_ts=-1,
        )

    def _transform(self, nodal_chi: Array) -> None:
        r"""
        Transform orientation-dependent results between frames using the nodal chi transformation matrices.
        :param nodal_chi: Nodal rotations represented as chi transformation matrices, [n_nodes_, 6, 6]
        """
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_follower)
            )
        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_dead)
            )
        if self.f_ext_aero is not None:
            self.f_ext_aero = self.f_ext_aero.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_aero)
            )
        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_grav)
            )
        self.f_int = self.f_int.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_int)
        )
        self.f_res = self.f_int.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_res)
        )

    def to_global(self) -> None:
        """Convert local structure_dv results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes_, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure_dv results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes_, 6, 6]

        self._transform(nodal_chi)

    def plot(self, directory: os.PathLike | str, n_interp: int = 0) -> Path:
        r"""
        Plot beam results to VTK files in the specified directory.
        :param directory: Path to write files to.
        :param n_interp: Number of interpolation points to add between each element for smoother visualization.
        """
        return self.to_dynamic().plot(directory, n_interp)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "conn", "o0", "prescribed_dofs"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "hg",
            "d",
            "eps",
            "f_ext_follower",
            "f_ext_dead",
            "f_ext_aero",
            "f_grav",
            "f_int",
            "f_res",
            "local",
        )


class DynamicStructureSnapshot:
    """Object to hold the full state and forces of a dynamic structure_dv analysis timestep."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        o0: Array,
        d: Array,
        eps: Array,
        v: Array,
        v_dot: Array,
        a: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_ext_aero: Optional[Array],
        f_grav: Optional[Array],
        f_int: Array,
        f_iner: Array,
        f_res: Array,
        t: Array,
        i_ts: int,
        prescribed_dofs: Optional[Array] = None,
        local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes_, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.v: Array = v  # [n_nodes_, 6]
        self.v_dot: Array = v_dot  # [n_nodes_, 6]
        self.a: Array = a  # [n_nodes_, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes_, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes_, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_nodes_, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes_, 6]
        self.f_int: Array = f_int  # [n_nodes_, 6]
        self.f_iner: Array = f_iner  # [n_nodes_, 6]
        self.f_res: Array = f_res  # [n_nodes_, 6]
        self.t: Array = t  # Scalar time value
        self.i_ts: int = i_ts  # Time step index
        self.prescribed_dofs: Optional[Array] = prescribed_dofs
        self.local: bool = local

    def to_static(self) -> StaticStructure:
        """Convert dynamic structure_dv snapshot results to a static structure_dv, dropping velocity/acceleration dependent
        entries."""
        return StaticStructure(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_res=self.f_res,
            prescribed_dofs=self.prescribed_dofs,
        )

    def _transform(self, nodal_chi: Array) -> None:
        r"""
        Transform orientation-dependent results between frames using the nodal chi transformation matrices.
        :param nodal_chi: Nodal rotations represented as chi transformation matrices, [n_nodes_, 6, 6]
        """
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_follower)
            )
        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_dead)
            )
        if self.f_ext_aero is not None:
            self.f_ext_aero = self.f_ext_aero.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_ext_aero)
            )
        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_grav)
            )
        self.f_int = self.f_int.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_int)
        )
        self.f_iner = self.f_iner.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_iner)
        )
        self.f_res = self.f_iner.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.f_res)
        )
        self.v = self.v.at[...].set(jnp.einsum("ijk,ik->ij", nodal_chi, self.v))
        self.v_dot = self.v_dot.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.v_dot)
        )
        self.a = self.a.at[...].set(jnp.einsum("ijk,ik->ij", nodal_chi, self.a))

    def to_global(self) -> None:
        """Convert local structure_dv results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes_, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure_dv results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes_, 6, 6]
        self._transform(nodal_chi)

    def plot(self, directory: os.PathLike | str, n_interp: int = 0) -> Path:
        r"""
        Plot beam results to VTK files in the specified directory. Other beam object types will first convert to a
        DynmaicStructureSnapshot before plotting.
        :param directory: Path to write files to.
        :param n_interp: Number of interpolation points to add between each element for smoother visualization.
        """

        # represent all vectors in the intertial frame
        data = deepcopy(
            self
        )  # prevent modification of original data when converting frames for plotting
        data.to_global()

        # vectors making up local frame rotation matrices
        local_x = data.hg[:, :3, 0]  # [n_nodes_, 3]
        local_y = data.hg[:, :3, 1]  # [n_nodes_, 3]
        local_z = data.hg[:, :3, 2]  # [n_nodes_, 3]

        # forcing data
        f_ext_follower = (
            data.f_ext_follower[:, :3] if data.f_ext_follower is not None else None
        )  # [n_nodes_, 3]
        m_ext_follower = (
            data.f_ext_follower[:, 3:] if data.f_ext_follower is not None else None
        )  # [n_nodes_, 3]
        f_ext_dead = (
            data.f_ext_dead[:, :3] if data.f_ext_dead is not None else None
        )  # [n_nodes_, 3]
        m_ext_dead = (
            data.f_ext_dead[:, 3:] if data.f_ext_dead is not None else None
        )  # [n_nodes_, 3]
        f_ext_aero = (
            data.f_ext_aero[:, :3] if data.f_ext_aero is not None else None
        )  # [n_nodes_, 3]
        m_ext_aero = (
            data.f_ext_aero[:, 3:] if data.f_ext_aero is not None else None
        )  # [n_nodes_, 3]
        f_ext_grav = (
            data.f_grav[:, :3] if data.f_grav is not None else None
        )  # [n_nodes_, 3]
        m_ext_grav = (
            data.f_grav[:, 3:] if data.f_grav is not None else None
        )  # [n_nodes_, 3]
        f_iner = data.f_iner[:, :3]  # [n_nodes_, 3]
        m_iner = data.f_iner[:, 3:]  # [n_nodes_,
        f_int = data.f_int[:, :3]  # [n_nodes_, 3]
        m_int = data.f_int[:, 3:]  # [n_nodes_, 3]
        f_res = data.f_res[:, :3]  # [n_nodes_, 3]
        m_res = data.f_res[:, 3:]  # [n_nodes_, 3]

        # velocity and acceleration data
        v_lin = data.v[:, :3]  # [n_nodes_, 3]
        v_ang = data.v[:, 3:]  # [n_nodes_, 3]
        v_dot_lin = data.v_dot[:, :3]  # [n_nodes_, 3]
        v_dot_ang = data.v_dot[:, 3:]  # [n_nodes_,

        # beam strain data
        eps_lin = data.eps[:, :3]  # [n_elem, 3]
        eps_ang = data.eps[:, 3:]  # [n_elem, 3]

        # node and element numbers for plotting
        node_num = jnp.arange(data.hg.shape[0])  # [n_nodes_]
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
            "f_ext_aero": f_ext_aero,
            "m_ext_aero": m_ext_aero,
            "f_ext_grav": f_ext_grav,
            "m_ext_grav": m_ext_grav,
            "f_iner": f_iner,
            "m_iner": m_iner,
            "f_int": f_int,
            "m_int": m_int,
            "f_res": f_res,
            "m_res": m_res,
            "v_linear": v_lin,
            "v_angular": v_ang,
            "v_dot_linear": v_dot_lin,
            "v_dot_angular": v_dot_ang,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {"eps_linear": eps_lin, "eps_angular": eps_ang}

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(directory).joinpath(
            "beam"
        )  # default file name for beam objects is "beam"
        return plot_beam_to_vtk(
            hg=data.hg,
            conn=data.conn,
            o0=data.o0,
            n_interp=n_interp,
            filename=file_name,
            i_ts=data.i_ts,
            node_scalar_data=node_scalar_data,
            node_vector_data=node_vector_data,
            cell_scalar_data=cell_scalar_data,
            cell_vector_data=cell_vector_data,
        )


@_make_pytree
class DynamicStructure:
    """Object to hold the full state and forces of a dynamic structure_dv analysis across multiple timesteps."""

    def __init__(
        self,
        hg: Array,
        conn: Array,
        o0: Array,
        d: Array,
        eps: Array,
        v: Array,
        v_dot: Array,
        a: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_ext_aero: Optional[Array],
        f_grav: Optional[Array],
        f_int: Array,
        f_iner: Array,
        f_res: Array,
        t: Array,
        n_tstep: int,
        prescribed_dofs: Optional[Array],
    ):
        self.hg: Array = hg  # [n_tstep, n_nodes_, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_tstep, n_elem, 6]
        self.eps: Array = eps  # [n_tstep, n_elem, 6]
        self.v: Array = v  # [n_tstep, n_nodes_, 6]
        self.v_dot: Array = v_dot  # [n_tstep, n_nodes_, 6]
        self.a: Array = a  # [n_tstep, n_nodes_, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_tstep, n_nodes_, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_tstep, n_nodes_, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_tstep, n_nodes_, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_tstep, n_nodes_, 6]
        self.f_int: Array = f_int  # [n_tstep, n_nodes_, 6]
        self.f_iner: Array = f_iner  # [n_tstep, n_nodes_, 6]
        self.f_res: Array = f_res  # [n_tstep, n_nodes_, 6]
        self.t: Array = t  # [n_tstep]
        self.n_tstep: int = n_tstep
        self.prescribed_dofs: Optional[Array] = prescribed_dofs
        self.local: bool = True

    def to_static(self, i_ts: int) -> StaticStructure:
        """Extract static structure_dv results at a specific time index."""
        return StaticStructure(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            o0=self.o0,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            f_ext_follower=self.f_ext_follower[i_ts, ...]
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=self.f_ext_dead[i_ts, ...]
            if self.f_ext_dead is not None
            else None,
            f_ext_aero=self.f_ext_aero[i_ts, ...]
            if self.f_ext_aero is not None
            else None,
            f_grav=self.f_grav[i_ts, ...] if self.f_grav is not None else None,
            f_int=self.f_int[i_ts, ...],
            f_res=self.f_res[i_ts, ...],
            prescribed_dofs=self.prescribed_dofs,
        )

    def __getitem__(self, i_ts: int) -> DynamicStructureSnapshot:
        """Extract dynamic structure_dv snapshot at a specific time index."""
        return DynamicStructureSnapshot(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            o0=self.o0,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            v=self.v[i_ts, ...],
            v_dot=self.v_dot[i_ts, ...],
            a=self.a[i_ts, ...],
            f_ext_follower=self.f_ext_follower[i_ts, ...]
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=self.f_ext_dead[i_ts, ...]
            if self.f_ext_dead is not None
            else None,
            f_ext_aero=self.f_ext_aero[i_ts, ...]
            if self.f_ext_aero is not None
            else None,
            f_grav=self.f_grav[i_ts, ...] if self.f_grav is not None else None,
            f_iner=self.f_iner[i_ts, ...],
            f_int=self.f_int[i_ts, ...],
            f_res=self.f_res[i_ts, ...],
            t=self.t[i_ts],
            i_ts=i_ts,
            prescribed_dofs=self.prescribed_dofs,
        )

    @classmethod
    def initialise(
        cls,
        initial_snapshot: DynamicStructureSnapshot,
        t: Array,
        prescribed_dofs: Optional[Array],
    ) -> DynamicStructure:
        r"""
        Initialise a DynamicStructure object given an initial snapshot and number of time steps.
        :param initial_snapshot: Snapshot at initial time step.
        :param t: Time step array, [n_tstep]
        :param prescribed_dofs: Array of prescribed DOF indices.
        :return: DynamicStructure object with arrays initialised to zero except for the first time step.
        """
        n_node = initial_snapshot.hg.shape[0]
        n_elem = initial_snapshot.d.shape[0]
        n_tstep = t.shape[0]

        hg = jnp.zeros((n_tstep, n_node, 4, 4)).at[0, ...].set(initial_snapshot.hg)
        conn = initial_snapshot.conn
        o0 = initial_snapshot.o0
        d = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d)
        eps = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.eps)
        v = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v)
        v_dot = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v_dot)
        a = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.a)
        f_ext_follower = (
            jnp.zeros((n_tstep, n_node, 6))
            .at[0, ...]
            .set(initial_snapshot.f_ext_follower)
        )
        f_ext_dead = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_ext_dead)
        )
        f_ext_aero = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_ext_aero)
        )
        f_grav = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_grav)
            if initial_snapshot.f_grav is not None
            else None
        )
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        f_iner = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_iner)
        f_res = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_res)
        return cls(
            hg=hg,
            conn=conn,
            o0=o0,
            d=d,
            eps=eps,
            v=v,
            v_dot=v_dot,
            a=a,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            f_ext_aero=f_ext_aero,
            f_grav=f_grav,
            f_int=f_int,
            f_iner=f_iner,
            f_res=f_res,
            t=t,
            n_tstep=n_tstep,
            prescribed_dofs=prescribed_dofs,
        )

    def _transform(self, nodal_chi: Array) -> None:
        r"""
        Transform orientation-dependent results between frames using the nodal chi transformation matrices.
        :param nodal_chi: Nodal rotations represented as chi transformation matrices, [n_tstep, n_nodes_, 6, 6]
        """
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                jnp.einsum("hijk,hik->hij", nodal_chi, self.f_ext_follower[...])
            )
        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(
                jnp.einsum("hijk,hik->hij", nodal_chi, self.f_ext_dead)
            )
        if self.f_ext_aero is not None:
            self.f_ext_aero = self.f_ext_aero.at[...].set(
                jnp.einsum("hijk,hik->hij", nodal_chi, self.f_ext_aero)
            )
        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(
                jnp.einsum("hijk,hik->hij", nodal_chi, self.f_grav)
            )
        self.f_int = self.f_int.at[...].set(
            jnp.einsum("hijk,hik->hij", nodal_chi, self.f_int)
        )
        self.f_iner = self.f_iner.at[...].set(
            jnp.einsum("hijk,hik->hij", nodal_chi, self.f_iner)
        )
        self.f_res = self.f_iner.at[...].set(
            jnp.einsum("hijk,hik->hij", nodal_chi, self.f_res)
        )
        self.v = self.v.at[...].set(jnp.einsum("hijk,hik->hij", nodal_chi, self.v))
        self.v_dot = self.v_dot.at[...].set(
            jnp.einsum("hijk,hik->hij", nodal_chi, self.v_dot)
        )
        self.a = self.a.at[...].set(jnp.einsum("hijk,hik->hij", nodal_chi, self.a))

    def to_global(self) -> None:
        """Convert local structure_dv results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(vmap(chi, 0, 0), 1, 1)(
            self.hg[:, :, :3, :3]
        )  # [n_tstep, n_nodes_, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure_dv results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(
            vmap(chi, 0, 0),
            1,
        )(
            jnp.transpose(self.hg[:, :, :3, :3], (0, 1, 3, 2))
        )  # [n_tstep, n_nodes_, 6, 6]
        self._transform(nodal_chi)

    def plot(
        self,
        directory: os.PathLike,
        index: Optional[slice | Sequence[int] | int | Array] = None,
        n_interp: int = 0,
    ) -> Path:
        r"""
        Plot the beam for specified time steps to VTU files in the specified directory. Additionally, a PVD
        file is created to include time data for visualization in Paraview.
        :param directory: Path to write files to
        :param index: Single or multiple time step indices to plot. If None, all time steps are plotted.
        :param n_interp: Number of interpolation points to add between each element for smoother visualization.
        """
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index,)
        elif index is None:
            index_ = jnp.arange(self.n_tstep)
        else:
            raise TypeError("index must be a slices, sequence of ints, or Array")

        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for i_ts in index_:
            snapshot = self[i_ts]
            paths.append(snapshot.plot(directory, n_interp))

        return write_pvd(directory, "beam_dynamic_ts", paths, list(self.t[index_]))

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Get names of static attributes in dynamic beam
        """
        return (
            "conn",
            "o0",
            "n_tstep",
            "prescribed_dofs",
        )

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        r"""
        Get names of dynamic attributes in dynamic beam
        """
        return (
            "hg",
            "d",
            "eps",
            "v",
            "v_dot",
            "a",
            "f_ext_follower",
            "f_ext_dead",
            "f_ext_aero",
            "f_grav",
            "f_int",
            "f_iner",
            "f_res",
            "t",
            "local",
        )
