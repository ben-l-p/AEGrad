from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from jax import Array
from jax import numpy as jnp

from flapjax.algebra.se3 import hg_to_d, t_se3
from flapjax.utils.utils import make_pytree


class NodalConstraint(ABC):
    r"""
    Base class for a single-node structural constraint.
    """

    node_index: int  # node where the constraint is applied

    @abstractmethod
    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        r"""
        Compute the forcing residual contribution from the constraint
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param v: Node local velocity, ``(6, )``.
        :param i_ts: Time-step index.
        :return: Nodal force, ``(6, )``.
        """

    @abstractmethod
    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        r"""
        Effective stiffness contribution :math:`-\partial \mathbf{f_{res}}/\partial \boldsymbol{\varphi}`.
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param i_ts: Time-step index.
        :return: Nodal stiffness contribution, ``(6, 6)``.
        """

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        r"""
        Effective damping contribution :math:`-\partial \mathbf{f_{res}}/\partial \mathbf{v}`.
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param i_ts: Time-step index.
        :return: Nodal damping contribution, ``(6, 6)``. Default implementation returns zero damping.
        """
        return jnp.zeros((6, 6))

    def resolve_hg_ref(self, hg0: Array) -> None:
        r"""
        Hook called by :meth:`BaseBeamStructure.set_design_variables` once ``hg0`` is populated,
        so subclasses can default their reference frame to the node's initial pose. Default: no-op.
        :param hg0: Full nodal initial-frame array, ``(n_nodes, 4, 4)``.
        """


@make_pytree
class SpringDamper(NodalConstraint):
    r"""
    6-DOF spring-damper attached to a fixed reference frame.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_index", "_hg_ref_from_hg0")

    def __init__(
        self,
        node_index: int,
        k: Array,
        hg_ref: Array | None = None,
        c: Array | None = None,
    ) -> None:
        r"""
        :param node_index: Index of the node to attach to.
        :param k: Stiffness matrix, ``(6, 6)``.
        :param hg_ref: Reference SE(3) frame, ``(4, 4)``, or ``None`` to default
        to the node's initial pose.
        :param c: Damping matrix, ``(6, 6)`` or ``None`` for no damping.
        """
        self.node_index: int = int(node_index)
        self.k: Array = k
        # sentinel: hg_ref is always a (4, 4) Array so the pytree structure is stable.
        # The static flag records whether the beam should overwrite it with hg0[node].
        self._hg_ref_from_hg0: bool = hg_ref is None
        self.hg_ref: Array = jnp.eye(4) if hg_ref is None else hg_ref
        self.c: Array = jnp.zeros((6, 6)) if c is None else c

    def resolve_hg_ref(self, hg0: Array) -> None:
        if self._hg_ref_from_hg0:
            self.hg_ref = hg0[self.node_index]

    def _varphi(self, hg_i: Array) -> Array:
        return hg_to_d(self.hg_ref, hg_i)

    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        return -self.k @ self._varphi(hg) - self.c @ v

    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.k @ t_se3(self._varphi(hg))

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.c


@make_pytree
class Hinge(SpringDamper):
    r"""
    Single-node hinge with an arbitrary axis. This is only imposed as a soft constraint, so the hinge is not perfectly
    rigid. Considered as a special case of the spring-damper constraint.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_index", "_hg_ref_from_hg0")

    def __init__(
        self,
        node_index: int,
        axis: Array,
        hg_ref: Array | None = None,
        k_translation: float | Array = 1e10,
        k_perpendicular: float | Array = 1e10,
        k_axis: float | Array = 0.0,
        c_translation: float | Array = 0.0,
        c_perpendicular: float | Array = 0.0,
        c_axis: float | Array = 0.0,
    ) -> None:
        r"""
        :param node_index: Index of the hinged node.
        :param axis: Hinge axis direction, ``(3, )``, giving in the reference global coordinate system.
        :param hg_ref: Reference SE(3) frame, ``(4, 4)``, or ``None`` to default
        to the node's initial pose.
        :param k_translation: Translation-blocking stiffness, () | (3, ).
        :param k_perpendicular: Rotation-blocking stiffness for axes perpendicular to the hinge axis.
        :param k_axis: Torsional stiffness about the hinge axis.
        :param c_translation: Translation-blocking damping, () | (3, ).
        :param c_perpendicular: Rotation-blocking damping for axes perpendicular to the hinge axis.
        :param c_axis: Torsional damping about the hinge axis.
        """
        a = axis / jnp.linalg.norm(axis)  # normalised hinge axis
        p_axis = jnp.outer(a, a)
        p_perp = jnp.eye(3) - p_axis

        # stiffness contribution
        k = jnp.zeros((6, 6))
        k = k.at[:3, :3].set(k_translation * jnp.eye(3))
        k = k.at[3:, 3:].set(k_axis * p_axis + k_perpendicular * p_perp)

        # damping contribution
        c = jnp.zeros((6, 6))
        c = c.at[:3, :3].set(c_translation * jnp.eye(3))
        c = c.at[3:, 3:].set(c_axis * p_axis + c_perpendicular * p_perp)

        super().__init__(node_index=node_index, k=k, hg_ref=hg_ref, c=c)
        self.axis: Array = a


@make_pytree
class PrescribedMotion(NodalConstraint):
    r"""
    Prescribe the trajectory of a node along a reference SE(3) history. The node is driven through a 6-DOF spring-damper
    to the reference trajectory.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_index",)

    def __init__(
        self,
        node_index: int,
        k: Array,
        hg_ref_t: Array,
        c: Array | None = None,
    ) -> None:
        r"""
        :param node_index: Index of the node to drive.
        :param k: Tracking stiffness, ``(6, 6)``.
        :param hg_ref_t: Reference SE(3) trajectory, ``(n_tstep, 4, 4)``.
        :param c: Tracking damping, ``(6, 6)`` or ``None`` for no damping.
        """
        self.node_index: int = int(node_index)
        self.k: Array = k
        self.hg_ref_t: Array = hg_ref_t
        self.c: Array = jnp.zeros((6, 6)) if c is None else c

    def _varphi(self, hg_i: Array, i_ts: int) -> Array:
        return hg_to_d(self.hg_ref_t[i_ts], hg_i)

    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        return -self.k @ self._varphi(hg, i_ts) - self.c @ v

    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.k @ t_se3(self._varphi(hg, i_ts))

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.c
