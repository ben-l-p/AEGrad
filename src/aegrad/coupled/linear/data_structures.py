from dataclasses import dataclass
from typing import Optional

from jax import Array
from aegrad.algebra.array_utils import ArrayList
from aegrad.structure.linear.data_structures import BeamLinearResult
from aegrad.aero.linear.data_structures import AeroLinearResult


@dataclass
class AeroelasticInputUnflattened:
    nu_b: Optional[ArrayList]
    nu_w: Optional[ArrayList]
    f_ext_follower: Optional[Array]
    f_ext_dead: Optional[Array]


@dataclass
class AeroelasticStateUnflattened:
    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_b_nm1: Optional[ArrayList]
    zeta_w: Optional[ArrayList]
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
