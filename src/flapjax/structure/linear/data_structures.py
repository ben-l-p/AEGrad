from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from flapjax.structure import StructureCase


@dataclass
class BeamInputUnflattened:
    n_tstep: int
    f_ext: Array | None


@dataclass
class BeamStateUnflattened:
    q: Array
    q_dot: Array


@dataclass
class BeamOutputUnflattened:
    q: Array
    q_dot: Array


class BeamLinearResult:
    def __init__(
        self,
        reference: StructureCase,
        f_ext: Array | None,
        delta_q: Array,
        delta_q_dot: Array,
        hg: Array | None,
        t: Array,
    ) -> None:
        # system results, if simulated
        self.f_ext: Array | None = f_ext
        self.delta_q: Array = delta_q
        self.delta_q_dot: Array = delta_q_dot
        self._hg: Array | None = hg
        self.n_tstep: int = len(t)
        self.t: Array = t
        self.reference: StructureCase = reference

    @property
    def hg(self) -> Array:
        if self._hg is None:
            raise ValueError("hg is not available for modal output")
        return self._hg

    @property
    def x(self) -> Array:
        return self.hg[..., :3, 3]

    @property
    def rmat(self) -> Array:
        return self.hg[..., :3, :3]
