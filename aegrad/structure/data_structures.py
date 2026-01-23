from dataclasses import dataclass
from jax import Array
from typing import Optional


@dataclass
class StaticStructure:
    """Results of a static structure analysis."""

    hg: Array
    d: Array
    eps: Array
    f_ext_follower: Optional[Array]
    f_ext_dead: Optional[Array]
    f_int: Array
