from __future__ import annotations
from jax import numpy as jnp
from jax import Array
from typing import Optional


class StaticStructure:
    """Results of a static structure analysis."""

    def __init__(
        self,
        hg: Array,
        d: Array,
        eps: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        f_int: Array,
    ):
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.d: Array = d  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]

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
            d=self.d,
            d_dot=zero_d_dot,
            eps=self.eps,
            v=zero_v,
            v_dot=zero_v_dot,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_int=self.f_int,
            t=zero_time,
        )


class DynamicStructureSnapshot:
    """Snapshot of dynamic structure results at a specific time."""

    def __init__(
        self,
        hg: Array,
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
        self.hg: Array = hg  # [n_nodes, 4, 4]
        self.d: Array = d  # [n_elem, 6]
        self.d_dot: Array = d_dot  # [n_elem, 6]
        self.eps: Array = eps  # [n_elem, 6]
        self.v: Array = v  # [n_nodes, 6]
        self.v_dot: Array = v_dot  # [n_nodes, 6]
        self.f_ext_follower: Optional[Array] = f_ext_follower  # [n_nodes, 6]
        self.f_ext_dead: Optional[Array] = f_ext_dead  # [n_nodes, 6]
        self.f_int: Array = f_int  # [n_nodes, 6]
        self.t: Array = t  # Scalar time value

    def to_static(self) -> StaticStructure:
        """Extract static structure results from the snapshot."""
        return StaticStructure(
            hg=self.hg,
            d=self.d,
            eps=self.eps,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_int=self.f_int,
        )


class DynamicStructure:
    """Results of a dynamic structure analysis."""

    def __init__(
        self,
        hg: Array,
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
        d = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d)
        d_dot = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d_dot)
        eps = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.eps)
        v = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v)
        v_dot = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v_dot)
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        t = jnp.zeros((n_tstep,)).at[0].set(initial_snapshot.t)
        return cls(hg, d, d_dot, eps, v, v_dot, f_int, t)
