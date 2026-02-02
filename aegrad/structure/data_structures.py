from __future__ import annotations
from copy import deepcopy
from os import PathLike
from pathlib import Path

from jax import numpy as jnp
from jax import Array
from typing import Optional
from aegrad.utils import make_pytree
from typing import Sequence
from aegrad.print_output import warn, jax_print, VerbosityLevel
from aegrad.algebra.base import chi
from jax import vmap

from plotting.beam import plot_beam_to_vtk
from plotting.pvd import write_pvd


@make_pytree
class ConvergenceStatus:
    def __init__(
        self, vect: Array, abs_tol: Array, i_iter: Array, n_iter: Optional[Array]
    ):
        self.max_elem = jnp.abs(vect).max()

        self.converged: Array = (self.max_elem < abs_tol) & (i_iter > 0)
        self.final_iter: Array = (
            (i_iter >= n_iter) if n_iter is not None else jnp.zeros((), dtype=bool)
        )
        self.has_nan: Array = jnp.isnan(vect).any()
        self.i_iter: Array = i_iter
        self.n_iter: Array = n_iter
        self.abs_tol: Array = abs_tol

    def get_status(self) -> Array:
        """Get overall convergence status."""
        return self.converged | self.has_nan | self.final_iter

    def print_message(self, i_ts: Optional[int], i_load_step: Optional[int]) -> None:
        """Print convergence message based on status."""

        if i_ts is None:
            jax_print(
                "Load step: {i_load_step:<4} | Iterations: {i_iter:<3} | Converged: {conv:1} | Residual: {res:.03e}",
                verbose_level=VerbosityLevel.NORMAL,
                i_load_step=i_load_step,
                i_iter=self.i_iter,
                conv=self.converged,
                res=self.max_elem,
            )
        else:
            jax_print(
                "Time step: {i_ts:<4} | Iterations: {i_iter:<3} | Converged: {conv:1} | Residual: {res:.03e}",
                verbose_level=VerbosityLevel.NORMAL,
                i_ts=i_ts,
                i_iter=self.i_iter,
                conv=self.converged,
                res=self.max_elem,
            )

    @staticmethod
    def _static_names() -> Sequence[str]:
        return "n_iter", "abs_tol"

    @staticmethod
    def _dynamic_names() -> Sequence[str]:
        return "converged", "final_iter", "has_nan", "i_iter", "max_elem"


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
        f_grav: Optional[Array],
        f_int: Array,
        local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.local: bool = local

    def to_dynamic(self) -> DynamicStructureSnapshot:
        """Convert static structure results to a dynamic structure snapshot with zero velocities."""
        n_nodes = self.hg.shape[0]
        zero_v = jnp.zeros((n_nodes, 6))
        zero_v_dot = jnp.zeros((n_nodes, 6))
        zero_a = jnp.zeros((n_nodes, 6))
        zero_time = jnp.array(0.0)
        zero_f_iner = jnp.zeros((n_nodes, 6))

        return DynamicStructureSnapshot(
            hg=self.hg,
            conn=self.conn,
            d=self.d,
            eps=self.eps,
            v=zero_v,
            v_dot=zero_v_dot,
            a=zero_a,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_iner=zero_f_iner,
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
        if self.f_grav is not None:
            self.f_grav = self.f_grav.at[...].set(
                jnp.einsum("ijk,ik->ij", nodal_chi, self.f_grav)
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

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes, 6, 6]

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
        eps: Array,
        v: Array,
        v_dot: Array,
        a: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_grav: Optional[Array],
        f_int: Array,
        f_iner: Array,
        t: Array,
        i_ts: int,
        local: bool = True,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.v: Array = v  # [n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_nodes, 6]
        self.a: Array = a  # [n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.f_iner: Array = f_iner  # [n_nodes, 6]
        self.t: Array = t  # Scalar time value
        self.i_ts: int = i_ts  # Time step index
        self.local: bool = local

    def to_static(self) -> StaticStructure:
        """Extract static structure results from the snapshot."""
        return StaticStructure(
            hg=self.hg,
            conn=self.conn,
            d=self.d,
            eps=self.eps,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_grav=self.f_grav,
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
        self.v = self.v.at[...].set(jnp.einsum("ijk,ik->ij", nodal_chi, self.v))
        self.v_dot = self.v_dot.at[...].set(
            jnp.einsum("ijk,ik->ij", nodal_chi, self.v_dot)
        )
        # TODO: should pseudoacceleration also be transformed?

    def to_global(self) -> None:
        """Convert local structure results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        else:
            self.local = False

        nodal_chi = vmap(chi, 0, 0)(self.hg[:, :3, :3])  # [n_nodes, 6, 6]
        self._transform(nodal_chi)

    def to_local(self) -> None:
        """Convert global structure results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        else:
            self.local = True

        nodal_chi = vmap(chi, 0, 0)(
            jnp.transpose(self.hg[:, :3, :3], (0, 2, 1))
        )  # [n_nodes, 6, 6]
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
        f_ext_grav = (
            data.f_grav[:, :3] if data.f_grav is not None else None
        )  # [n_nodes, 3]
        m_ext_grav = (
            data.f_grav[:, 3:] if data.f_grav is not None else None
        )  # [n_nodes, 3]
        f_iner = data.f_iner[:, :3]  # [n_nodes, 3]
        m_iner = data.f_iner[:, 3:]  # [n_nodes,
        f_int = data.f_int[:, :3]  # [n_nodes, 3]
        m_int = data.f_int[:, 3:]  # [n_nodes, 3]

        v_lin = data.v[:, :3]  # [n_nodes, 3]
        v_ang = data.v[:, 3:]  # [n_nodes, 3]
        v_dot_lin = data.v_dot[:, :3]  # [n_nodes, 3]
        v_dot_ang = data.v_dot[:, 3:]  # [n_nodes,
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
            "f_ext_grav": f_ext_grav,
            "m_ext_grav": m_ext_grav,
            "f_iner": f_iner,
            "m_iner": m_iner,
            "f_int": f_int,
            "m_int": m_int,
            "v_linear": v_lin,
            "v_angular": v_ang,
            "v_dot_linear": v_dot_lin,
            "v_dot_angular": v_dot_ang,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {"eps_linear": eps_lin, "eps_angular": eps_ang}

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(directory).joinpath("beam")
        return plot_beam_to_vtk(
            coords,
            data.conn,
            file_name,
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
        eps: Array,
        v: Array,
        v_dot: Array,
        a: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_grav: Optional[Array],
        f_int: Array,
        f_iner: Array,
        t: Array,
        n_tstep: int,
    ):
        self.hg: Array = hg  # [n_tstep, n_nodes, 4, 4]
        self.conn: Array = conn  # [n_elem, 2]
        self.d: Array = d  # [n_tstep, n_elem, 6]
        self.eps: Array = eps  # [n_tstep, n_elem, 6]
        self.v: Array = v  # [n_tstep, n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_tstep, n_nodes, 6]
        self.a: Array = a  # [n_tstep, n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_tstep, n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_tstep, n_nodes, 6]
        self.f_grav: Optional[Array] = f_grav  # [n_tstep, n_nodes, 6]
        self.f_int: Array = f_int  # [n_tstep, n_nodes, 6]
        self.f_iner: Array = f_iner  # [n_tstep, n_nodes, 6]
        self.t: Array = t  # [n_tstep]
        self.n_tstep: int = n_tstep

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
            f_grav=self.f_grav[i_ts, ...] if self.f_grav is not None else None,
            f_int=self.f_int[i_ts, ...],
        )

    def __getitem__(self, i_ts: int) -> DynamicStructureSnapshot:
        """Extract dynamic structure snapshot at a specific time index."""
        return DynamicStructureSnapshot(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
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
            f_grav=self.f_grav[i_ts, ...] if self.f_grav is not None else None,
            f_iner=self.f_iner[i_ts, ...],
            f_int=self.f_int[i_ts, ...],
            t=self.t[i_ts],
            i_ts=i_ts,
        )

    @classmethod
    def initialise(
        cls, initial_snapshot: DynamicStructureSnapshot, t: Array
    ) -> DynamicStructure:
        r"""
        Initialise a DynamicStructure object given an initial snapshot and number of time steps.
        :param initial_snapshot: Snapshot at initial time step.
        :param t: Time step array, [n_tstep]
        :return: DynamicStructure object with arrays initialised to zero except for the first time step.
        """
        n_node = initial_snapshot.hg.shape[0]
        n_elem = initial_snapshot.d.shape[0]
        n_tstep = t.shape[0]

        hg = jnp.zeros((n_tstep, n_node, 4, 4)).at[0, ...].set(initial_snapshot.hg)
        conn = initial_snapshot.conn
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
        f_grav = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_grav)
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        f_iner = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_iner)
        return cls(
            hg=hg,
            conn=conn,
            d=d,
            eps=eps,
            v=v,
            v_dot=v_dot,
            a=a,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            f_grav=f_grav,
            f_int=f_int,
            f_iner=f_iner,
            t=t,
            n_tstep=n_tstep,
        )

    def plot(
        self,
        directory: PathLike,
        index: Optional[slice | Sequence[int] | int | Array] = None,
    ) -> Path:
        r"""
        Plot the beam for specified time steps to VTU files in the specified directory. Additionally, a PVD
        file is created to allow easy loading of all time steps in Paraview.
        :param directory: Path to write files to
        :param index: Single or multiple time step indices to plot. If None, all time steps are plotted

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
            paths.append(snapshot.plot(directory))

        return write_pvd(directory, "beam_dynamic_ts", paths, list(self.t[index_]))

    @staticmethod
    def _static_names() -> Sequence[str]:
        r"""
        Get names of static attributes in dynamic beam
        """
        return "conn", "n_tstep"

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
            "f_grav",
            "f_int",
            "f_iner",
            "t",
        )
