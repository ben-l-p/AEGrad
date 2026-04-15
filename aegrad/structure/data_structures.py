from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Optional, Sequence, overload
from dataclasses import dataclass

from jax import numpy as jnp, Array
from jax import vmap

from utils.print_utils import warn
from algebra.base import chi
from plotting.beam import plot_beam_to_vtk
from plotting.pvd import write_pvd
from structure.utils import transform_nodal_vect
from utils.utils import make_pytree, index_to_arr
from structure.gradients.data_structures import StructureFullStates


@dataclass
class OptionalJacobians:
    d_f_ext_dead_d_n: bool = False  # stiffness from dead loads
    d_f_grav_d_n: bool = False  # stiffness from gravitational loads
    d_f_gyr_d_q_dot: bool = (
        False  # derivative of gyroscopic forces with respect to Q_dot
    )
    d_f_int_d_p_d: bool = False  # geometric stiffness


@make_pytree
class StaticStructure:
    """Object to hold the full state and forces of a static structure_dv analysis."""

    def __init__(
            self,
            hg: Array,
            conn: Array,
            o0: Array,
            d: Array,
            eps: Array,
            varphi: Array,
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            f_grav: Optional[Array],
            f_int: Array,
            f_elem: Array,
            f_res: Array,
            prescribed_dofs: Optional[Array],
            local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.varphi: Array = varphi  # [n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.f_elem: Array = f_elem  # [n_elem, 6]
        self.f_res: Array = f_res  # [n_nodes, 6]
        self.prescribed_dofs: Array = prescribed_dofs if prescribed_dofs is not None else jnp.zeros((0,), dtype=int)
        self.local: bool = local

    @overload
    def to_dynamic(self) -> DynamicStructureSnapshot:
        ...

    @overload
    def to_dynamic(self, t: Array) -> DynamicStructure:
        ...

    @overload
    def to_dynamic(self, t: None) -> DynamicStructureSnapshot:
        ...

    def to_dynamic(self, t: Optional[Array] = None) -> DynamicStructureSnapshot | DynamicStructure:
        """Convert static structure_dv results to a dynamic structure_dv initial_snapshot, with zeroed velocity/acceleration dependent
        entries."""
        n_nodes = self.hg.shape[0]
        zero_v = jnp.zeros((n_nodes, 6))
        zero_v_dot = jnp.zeros((n_nodes, 6))
        zero_a = jnp.zeros((n_nodes, 6))
        zero_time = jnp.array(0.0)
        zero_f_iner = jnp.zeros((n_nodes, 6))

        dyn_snapshot = DynamicStructureSnapshot(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            varphi=self.varphi,
            v=zero_v,
            v_dot=zero_v_dot,
            a=zero_a,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_elem=self.f_elem,
            f_iner_gyr=zero_f_iner,
            f_res=self.f_res,
            t=zero_time,
            i_ts=-1,
            prescribed_dofs=self.prescribed_dofs,
        )

        if t is None:
            return dyn_snapshot
        else:
            return DynamicStructure.initialise(initial_snapshot=dyn_snapshot, t=t,
                                               use_f_ext_aero=self.f_ext_aero is not None,
                                               use_f_ext_follower=self.f_ext_follower is not None,
                                               use_f_ext_dead=self.f_ext_dead is not None)

    def get_full_states(self) -> StructureFullStates:
        return StructureFullStates(v=None, v_dot=None, hg=self.hg,
                                   eps=self.eps, f_elem=self.f_elem)

    def _transform(self, rmat: Array) -> None:
        r"""
        Transform orientation-dependent results between frames using the nodal chi transformation matrices.
        :param rmat: Nodal rotations, [n_nodes, 3, 3]
        """
        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(transform_nodal_vect(self.f_ext_follower, rmat))

        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(transform_nodal_vect(self.f_ext_dead, rmat))

        if self.f_ext_aero is not None:
            self.f_ext_aero = self.f_ext_aero.at[...].set(transform_nodal_vect(self.f_ext_aero, rmat))

        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(transform_nodal_vect(self.f_grav, rmat))

        self.f_int = self.f_int.at[...].set(
            transform_nodal_vect(self.f_int, rmat)
        )
        self.f_res = self.f_int.at[...].set(
            transform_nodal_vect(self.f_res, rmat)
        )

    def to_global(self) -> None:
        """
        Convert local structure_dv results to global frame.
        """
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return

        self.local = False
        self._transform(rmat=self.hg[:, :3, :3])

    def to_local(self) -> None:
        """Convert global structure_dv results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        self._transform(rmat=jnp.transpose(self.hg[:, :3, :3], (0, 2, 1)))

    def plot(self, directory: os.PathLike | str, n_interp: int = 0) -> Path:
        r"""
        Plot beam results to VTK files in the specified directory.
        :param directory: Path to write files to.
        :param n_interp: Number of interpolation points to add between each element for smoother visualisation.
        """
        return self.to_dynamic(t=None).plot(directory, n_interp)

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "conn", "o0", "prescribed_dofs", "local"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return (
            "hg",
            "d",
            "eps",
            "varphi",
            "f_ext_follower",
            "f_ext_dead",
            "f_ext_aero",
            "f_grav",
            "f_int",
            "f_elem",
            "f_res",
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
            varphi: Array,
            v: Array,
            v_dot: Array,
            a: Array,
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            f_grav: Optional[Array],
            f_int: Array,
            f_elem: Array,
            f_iner_gyr: Array,
            f_res: Array,
            t: Array,
            i_ts: int,
            prescribed_dofs: Optional[Array],
            local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.varphi: Array = varphi  # [n_nodes, 6]
        self.v: Array = v  # [n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_nodes, 6]
        self.a: Array = a  # [n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.f_elem: Array = f_elem  # [n_elem, 6]
        self.f_iner: Array = f_iner_gyr  # [n_nodes, 6]
        self.f_res: Array = f_res  # [n_nodes, 6]
        self.t: Array = t  # Scalar time value
        self.i_ts: int = i_ts  # Time step index
        self.prescribed_dofs: Array = prescribed_dofs if prescribed_dofs is not None else jnp.zeros((0,), dtype=int)
        self.local: bool = local

    def to_static(self) -> StaticStructure:
        """Convert dynamic structure_dv initial_snapshot results to a static structure_dv, dropping velocity/acceleration dependent
        entries."""
        return StaticStructure(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            varphi=self.varphi,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_elem=self.f_elem,
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

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes, 6, 6]
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
        DynamicStructureSnapshot before plotting.
        :param directory: Path to write files to.
        :param n_interp: Number of interpolation points to add between each element for smoother visualisation.
        """

        # represent all vectors in the inertial frame
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
            "f_iner_gyr": f_iner,
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


@make_pytree
class DynamicStructure:
    """Object to hold the full state and forces of a dynamic structure_dv analysis across multiple timesteps."""

    def __init__(
            self,
            hg: Array,
            conn: Array,
            o0: Array,
            d: Array,
            eps: Array,
            varphi: Array,
            v: Array,
            v_dot: Array,
            a: Array,
            f_ext_follower: Optional[Array],
            f_ext_dead: Optional[Array],
            f_ext_aero: Optional[Array],
            f_grav: Optional[Array],
            f_int: Array,
            f_elem: Array,
            f_iner: Array,
            f_res: Array,
            t: Array,
            n_tstep: int,
            prescribed_dofs: Optional[Array],
    ):
        self.hg: Array = hg  # [n_tstep, n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.o0: Array = o0  # [n_elem, 3, 3]
        self.d: Array = d  # [n_tstep, n_elem, 6]
        self.eps: Array = eps  # [n_tstep, n_elem, 6]
        self.varphi: Array = varphi  # [n_tstep, n_nodes, 6]
        self.v: Array = v  # [n_tstep, n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_tstep, n_nodes, 6]
        self.a: Array = a  # [n_tstep, n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_tstep, n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_tstep, n_nodes, 6]
        self.f_ext_aero: Optional[Array] = f_ext_aero  # [n_tstep, n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_tstep, n_nodes, 6]
        self.f_int: Array = f_int  # [n_tstep, n_nodes, 6]
        self.f_elem: Array = f_elem  # [n_tstep, n_elem, 6]
        self.f_iner_gyr: Array = f_iner  # [n_tstep, n_nodes, 6]
        self.f_res: Array = f_res  # [n_tstep, n_nodes, 6]
        self.t: Array = t  # [n_tstep]
        self.n_tstep: int = n_tstep
        self.prescribed_dofs: Array = prescribed_dofs if prescribed_dofs is not None else jnp.zeros((0,), dtype=int)
        self.local: bool = True

    def to_static(self, i_ts: int) -> StaticStructure:
        """Extract static structure_dv results at a specific time index."""
        return StaticStructure(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            o0=self.o0,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            varphi=self.varphi[i_ts, ...],
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
            f_elem=self.f_elem[i_ts, ...],
            f_res=self.f_res[i_ts, ...],
            prescribed_dofs=self.prescribed_dofs,
        )

    def __getitem__(self, i_ts: int) -> DynamicStructureSnapshot:
        """Extract dynamic structure_dv initial_snapshot at a specific time index."""
        return DynamicStructureSnapshot(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            o0=self.o0,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            varphi=self.varphi[i_ts, ...],
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
            f_iner_gyr=self.f_iner_gyr[i_ts, ...],
            f_int=self.f_int[i_ts, ...],
            f_elem=self.f_elem[i_ts, ...],
            f_res=self.f_res[i_ts, ...],
            t=self.t[i_ts],
            i_ts=i_ts,
            prescribed_dofs=self.prescribed_dofs,
        )

    def get_minimal_states(self, i_ts: int | Array) -> StructureMinimalStates:
        return StructureMinimalStates(
            varphi=self.varphi[i_ts, ...],
            v=self.v[i_ts, ...],
            v_dot=self.v_dot[i_ts, ...],
            a=self.a[i_ts, ...],
            f_ext_aero=self.f_ext_aero[i_ts, ...] if self.f_ext_aero is not None else None,
        )

    def get_full_states(self, i_ts: int | Array) -> StructureFullStates:
        return StructureFullStates(v=self.v[i_ts, ...], v_dot=self.v_dot[i_ts, ...], hg=self.hg[i_ts, ...],
                                   eps=self.eps[i_ts, ...], f_elem=self.f_elem[i_ts, ...])

    @classmethod
    def initialise(
            cls,
            initial_snapshot: DynamicStructureSnapshot,
            t: Array,
            use_f_ext_follower: bool,
            use_f_ext_dead: bool,
            use_f_ext_aero: bool,
    ) -> DynamicStructure:
        r"""
        Initialise a DynamicStructure object given an initial initial_snapshot and number of time steps.
        :param initial_snapshot: Snapshot at initial time step.
        :param t: Time step array, [n_tstep]
        :param use_f_ext_follower: Whether to include follower force array
        :param use_f_ext_dead: Whether to include dead force array
        :param use_f_ext_aero: Whether to include aero force array
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
        varphi = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.varphi)
        v = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v)
        v_dot = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v_dot)
        a = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.a)

        if use_f_ext_follower:
            f_ext_follower = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_follower is not None:
                f_ext_follower = f_ext_follower.at[0, ...].set(initial_snapshot.f_ext_follower)
        else:
            f_ext_follower = None

        if use_f_ext_dead:
            f_ext_dead = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_follower is not None:
                f_ext_dead = f_ext_dead.at[0, ...].set(initial_snapshot.f_ext_dead)
        else:
            f_ext_dead = None

        if use_f_ext_aero:
            f_ext_aero = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_follower is not None:
                f_ext_aero = f_ext_aero.at[0, ...].set(initial_snapshot.f_ext_aero)
        else:
            f_ext_aero = None

        f_grav = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_grav)
            if initial_snapshot.f_grav is not None
            else None
        )
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        f_elem = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.f_elem)
        f_iner = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_iner)
        f_res = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_res)
        return cls(
            hg=hg,
            conn=conn,
            o0=o0,
            d=d,
            eps=eps,
            varphi=varphi,
            v=v,
            v_dot=v_dot,
            a=a,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            f_ext_aero=f_ext_aero,
            f_grav=f_grav,
            f_int=f_int,
            f_elem=f_elem,
            f_iner=f_iner,
            f_res=f_res,
            t=t,
            n_tstep=n_tstep,
            prescribed_dofs=initial_snapshot.prescribed_dofs,
        )

    def _transform(self, rmat: Array) -> None:
        r"""
        Transform orientation-dependent results between frames using the nodal chi transformation matrices.
        :param rmat: Nodal rotations, [n_tstep, n_nodes_, 3, 3]
        """

        if self.f_ext_follower is not None:
            self.f_ext_follower = self.f_ext_follower.at[...].set(
                transform_nodal_vect(vect=self.f_ext_follower, rmat=rmat))

        if self.f_ext_dead is not None:
            self.f_ext_dead = self.f_ext_dead.at[...].set(transform_nodal_vect(vect=self.f_ext_dead, rmat=rmat))

        if self.f_ext_aero is not None:
            self.f_ext_aero = self.f_ext_aero.at[...].set(transform_nodal_vect(vect=self.f_ext_aero, rmat=rmat))

        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(transform_nodal_vect(vect=self.f_grav, rmat=rmat))

        self.f_int = self.f_int.at[...].set(transform_nodal_vect(vect=self.f_int, rmat=rmat))
        self.f_iner_gyr = self.f_iner_gyr.at[...].set(transform_nodal_vect(vect=self.f_iner_gyr, rmat=rmat))
        self.f_res = self.f_res.at[...].set(transform_nodal_vect(vect=self.f_res, rmat=rmat))
        self.v = self.v.at[...].set(transform_nodal_vect(vect=self.v, rmat=rmat))
        self.v_dot = self.v_dot.at[...].set(transform_nodal_vect(vect=self.v_dot, rmat=rmat))
        self.a = self.a.at[...].set(transform_nodal_vect(vect=self.a, rmat=rmat))

    def to_global(self) -> None:
        """Convert local structure_dv results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        self._transform(rmat=self.hg[:, :, :3, :3])

    def to_local(self) -> None:
        """Convert global structure_dv results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        self._transform(rmat=jnp.transpose(self.hg[:, :, :3, :3], (0, 1, 3, 2)))

    def plot(
            self,
            directory: os.PathLike | str,
            index: Optional[slice | Sequence[int] | int | Array] = None,
            n_interp: int = 0,
    ) -> Path:
        r"""
        Plot the beam for specified time steps to VTU files in the specified directory. Additionally, a PVD
        file is created to include time data for visualisation in Paraview.
        :param directory: Path to write files to
        :param index: Single or multiple time step indices to plot. If None, all time steps are plotted.
        :param n_interp: Number of interpolation points to add between each element for smoother visualisation.
        """
        index_ = index_to_arr(index=index, n_entries=self.n_tstep)

        directory_path = Path(directory).resolve()
        directory_path.mkdir(parents=True, exist_ok=True)

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
            "varphi",
            "v",
            "v_dot",
            "a",
            "f_ext_follower",
            "f_ext_dead",
            "f_ext_aero",
            "f_grav",
            "f_int",
            "f_elem",
            "f_iner_gyr",
            "f_res",
            "t",
            "local",
        )


@make_pytree
class StructureMinimalStates:
    def __init__(
            self,
            varphi: Optional[Array],
            v: Array,
            v_dot: Array,
            a: Array,
            f_ext_aero: Optional[Array] = None,
    ):
        self._varphi: Optional[Array] = varphi
        self.v: Array = v
        self.v_dot: Array = v_dot
        self.a: Array = a
        self.f_ext_aero: Optional[Array] = f_ext_aero

    @property
    def varphi(self) -> Array:
        if self._varphi is None:
            raise ValueError("varphi is None")
        return self._varphi

    @varphi.setter
    def varphi(self, varphi: Array) -> None:
        self._varphi = varphi

    @classmethod
    def from_mat(cls, stacked_mat: Array) -> StructureMinimalStates:
        return StructureMinimalStates(*stacked_mat.reshape(stacked_mat.shape[0], -1, 6))

    def to_mat(self) -> Array:
        out = jnp.stack(
            (self.varphi, self.v, self.v_dot, self.a), 0
        )  # [4, n_nodes, 6]

        if self.f_ext_aero is not None:
            out = jnp.concatenate((out, self.f_ext_aero[None, ...]), 0)
        return out

    def ravel(self) -> Array:
        return self.to_mat().ravel()

    @property
    def n_states(self) -> int:
        return self.to_mat().size

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "_varphi", "v", "v_dot", "a", "f_ext_aero"

    @staticmethod
    def _static_names() -> Sequence[str]:
        return ()
