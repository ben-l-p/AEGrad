from dataclasses import dataclass

from jax import Array

from flapjax.aero.linear.data_structures import AeroLinearResult
from flapjax.algebra.array_utils import ArrayList
from flapjax.structure.linear.data_structures import BeamLinearResult


@dataclass
class AeroelasticInputUnflattened:
    nu_b: ArrayList | None
    nu_w: ArrayList | None
    f_ext: Array | None


@dataclass
class AeroelasticStateUnflattened:
    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_b_nm1: ArrayList | None
    zeta_w: ArrayList | None
    q: Array
    q_dot: Array


@dataclass
class AeroelasticOutputUnflattened:
    q: Array
    q_dot: Array


@dataclass
class AeroelasticLinearResult:
    aero: AeroLinearResult
    structure: BeamLinearResult
