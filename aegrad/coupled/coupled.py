from typing import Optional, Sequence

from jax import Array

from aegrad.algebra.array_utils import ArrayList
from aegrad.aero import UVLM
from aegrad.aero.flowfields import FlowField
from aegrad.structure import BeamStructure


class CoupledSystem:
    def __init__(self, structure: BeamStructure, aero: UVLM):
        self.structure: BeamStructure = structure
        self.aero: UVLM = aero

    def set_design_variables(
        self,
        coords: Array,
        k_cs: Array,
        m_cs: Array,
        m_lumped: Array,
        dt: float | Array,
        flowfield: FlowField,
        delta_w: Optional[Sequence[Array] | Array],
        x0_aero: ArrayList | Sequence[Array] | Array,
    ):
        self.structure.set_design_variables(
            coords=coords, k_cs=k_cs, m_cs=m_cs, m_lumped=m_lumped
        )
        self.aero.set_design_variables(
            dt=dt,
            flowfield=flowfield,
            delta_w=delta_w,
            x0_aero=x0_aero,
            hg0=self.structure.hg0,
        )

    def static_solve(
        self,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        prescribed_dofs: Sequence[int] | Array | slice | int,
        t: float | Array = 0.0,
        load_steps: int = 1,
        relaxation_factor: float = 1.0,
        max_n_fsi_iter: Optional[int] = 10,
        horseshoe: bool = False,
    ):

        hg = self.structure.hg0

        self.aero.solve_static(t=t, hg=hg, horseshoe=horseshoe)

        # f_aero = self.aero.project_forcing_to_beam(
        #     i_ts=i_ts, include_unsteady=False
        # )  # [n_nodes_, 6]
        #
        # result = self.structure.static_solve(
        #     f_ext_follower=f_ext_follower,
        #     f_ext_dead=f_ext_dead + f_aero,
        #     prescribed_dofs=prescribed_dofs,
        #     load_steps=load_steps,
        #     relaxation_factor=relaxation_factor,
        # )

    def dynamic_solve(self):
        pass
