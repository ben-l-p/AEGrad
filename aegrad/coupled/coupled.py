from typing import Optional, Sequence

import jax
from jax import Array, vmap

from aegrad.algebra.se3 import hg_to_d
from aegrad.algebra.array_utils import ArrayList
from aegrad.aero.uvlm import UVLM
from aegrad.aero.flowfields import FlowField
from aegrad.structure import BeamStructure
from aero.data_structures import DynamicAeroCase
from data_structures import ConvergenceSettings, ConvergenceStatus
from aegrad.coupled import StaticAeroelastic, DynamicAeroelastic
from aegrad.print_utils import warn_if_32_bit
from structure import StaticStructure
from aegrad.print_utils import VerbosityLevel


class BaseCoupledAeroelastic:
    def __init__(
        self,
        structure: BeamStructure,
        aero: UVLM,
        fsi_convergence_settings: Optional[ConvergenceSettings] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ):
        self.structure: BeamStructure = structure
        self.aero: UVLM = aero
        if fsi_convergence_settings is None:
            self.fsi_convergence_settings: ConvergenceSettings = ConvergenceSettings()
        else:
            self.fsi_convergence_settings = fsi_convergence_settings
        self.verbosity = verbosity

    def set_design_variables(
        self,
        coords: Array,
        k_cs: Array,
        m_cs: Optional[Array],
        m_lumped: Optional[Array],
        dt: float | Array,
        flowfield: FlowField,
        delta_w: Optional[Sequence[Array] | Array],
        x0_aero: ArrayList | Sequence[Array] | Array,
        *,
        remove_checks: bool = False,
    ):
        self.structure.set_design_variables(
            coords=coords,
            k_cs=k_cs,
            m_cs=m_cs,
            m_lumped=m_lumped,
            remove_checks=remove_checks,
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
        horseshoe: bool = False,
    ) -> StaticAeroelastic:

        warn_if_32_bit()

        def _convergence_loop(
            converge_status_: ConvergenceStatus,
            struct_case_n: StaticStructure,
            aero_case_n: DynamicAeroCase,
        ) -> tuple[ConvergenceStatus, StaticStructure, DynamicAeroCase]:

            f_aero_n = aero_case_n.project_forcing_to_beam(
                i_ts=0,
                rmat=struct_case_n.hg[:, :3, :3],
                x0_aero=self.aero.x0_b,
                include_unsteady=False,
            )  # [n_nodes_, 6]

            struct_case_np1 = self.structure.static_solve(
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                f_ext_aero=f_aero_n,
                prescribed_dofs=prescribed_dofs,
                load_steps=load_steps,
                relaxation_factor=relaxation_factor,
            )

            delta_n = vmap(hg_to_d)(
                struct_case_n.hg, struct_case_np1.hg
            )  # [n_nodes, 6]

            tot_n = vmap(hg_to_d)(struct_case_np1.hg, self.structure.hg0)

            delta_f = (
                struct_case_n.f_ext_aero - struct_case_np1.f_ext_aero
            )  # [n_nodes, 6]

            converge_status_.update(
                delta_disp=delta_n,
                total_disp=tot_n,
                delta_force=delta_f,
                total_force=struct_case_np1.f_ext_aero,
            )

            aero_case_np1 = self.aero.solve_static(
                t=t, hg=struct_case_np1.hg, horseshoe=horseshoe
            )

            if self.verbosity.value >= VerbosityLevel.NORMAL.value:
                converge_status_.print_fsi_message(None)

            return converge_status_, struct_case_np1, aero_case_np1

        convergence_status, struct_case, aero_case = jax.lax.while_loop(
            lambda args_: ~args_[0].get_status(),
            lambda args_: _convergence_loop(*args_),
            (
                ConvergenceStatus(self.fsi_convergence_settings),
                self.structure.reference_configuration(
                    use_f_grav=self.structure.use_gravity,
                    use_f_ext_dead=f_ext_dead is not None,
                    use_f_ext_follower=f_ext_follower is not None,
                    use_f_aero=True,
                    prescribed_dofs=prescribed_dofs,
                ),
                self.aero.solve_static(t=t, hg=self.structure.hg0, horseshoe=horseshoe),
            ),
        )

        return StaticAeroelastic(structure=struct_case, aero=aero_case)

    def dynamic_solve(self) -> DynamicAeroelastic:
        pass
