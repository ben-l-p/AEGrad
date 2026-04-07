from __future__ import annotations
from typing import Optional, Sequence

import jax
from jax import Array, vmap
from jax import numpy as jnp

from algebra.se3 import hg_to_d
from algebra.array_utils import ArrayList
from aero.uvlm import UVLM
from aero.flowfields import FlowField
from structure import BeamStructure
from aero.data_structures import DynamicAeroCase
from coupled.gradients.data_structures import AeroelasticDesignVariables
from data_structures import ConvergenceSettings, ConvergenceStatus
from coupled.data_structures import StaticAeroelastic, DynamicAeroelastic, DynamicAeroelasticSnapshot
from print_utils import warn_if_32_bit
from structure import StaticStructure
from print_utils import VerbosityLevel
from structure.time_integration import TimeIntregrator


class BaseCoupledAeroelastic:
    def __init__(
            self,
            structure: BeamStructure,
            aero: UVLM,
            fsi_convergence_settings: ConvergenceSettings = ConvergenceSettings(max_n_iter=25,
                                                                                rel_disp_tol=1e-3,
                                                                                abs_disp_tol=1e-5,
                                                                                rel_force_tol=1e-3,
                                                                                abs_force_tol=1e-5),
            verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ):
        self.structure: BeamStructure = structure
        self.aero: UVLM = aero
        self.fsi_convergence_settings: ConvergenceSettings = fsi_convergence_settings
        self.verbosity = verbosity

    def set_design_variables(
            self,
            coords: Array,
            k_cs: Array,
            m_cs: Optional[Array],
            m_lumped: Optional[Array],
            dt: float | Array,
            flowfield: FlowField,
            delta_w: Optional[Sequence[Optional[Array]] | Optional[Array]],
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

    def case_from_dv(self, dv: AeroelasticDesignVariables) -> BaseCoupledAeroelastic:
        return BaseCoupledAeroelastic(structure=self.structure.case_from_dv(dv.structure),
                                      aero=self.aero.case_from_dv(dv.aero), )

    def get_design_variables(self, case: StaticAeroelastic | DynamicAeroelastic) -> AeroelasticDesignVariables:
        return AeroelasticDesignVariables(
            structure_dv=self.structure.get_design_variables(struct_case=case.structure),
            aero_dv=self.aero.get_design_variables())

    def reference_configuration(self,
                                prescribed_dofs: Optional[Array],
                                use_f_ext_follower: bool = False,
                                use_f_ext_dead: bool = False,
                                use_f_ext_aero: bool = False,
                                t_init: float | Array = 0.0,
                                ) -> StaticAeroelastic:
        r"""
        Obtain the static aeroelastic object describing the undeformed wing.
        :param prescribed_dofs: Prescribed dofs for the structure
        :param use_f_ext_follower: If true, allocate an array for follower forces.
        :param use_f_ext_dead: If true, allocate an array for dead forces.
        :param use_f_ext_aero: If true, allocate an array for aero forces.
        :param t_init: Initial time
        :return: Static aeroelastic object for undeformed wing
        """
        return StaticAeroelastic(
            structure=self.structure.reference_configuration(
                use_f_grav=self.structure.use_gravity,
                use_f_ext_dead=use_f_ext_dead,
                use_f_ext_follower=use_f_ext_follower,
                use_f_aero=use_f_ext_aero,
                prescribed_dofs=prescribed_dofs,
            ),
            aero=self.aero.solve_static(t=t_init, hg=self.structure.hg0, horseshoe=False),
        )

    def static_solve(
            self,
            prescribed_dofs: Sequence[int] | Array | slice | int | None,
            f_ext_follower: Optional[Array] = None,
            f_ext_dead: Optional[Array] = None,
            t: float | Array = 0.0,
            load_steps: int = 1,
            relaxation_factor: float = 1.0,
            horseshoe: bool = False,
    ) -> StaticAeroelastic:

        warn_if_32_bit()

        prescribed_dofs: Array = self.structure.make_prescribed_dofs_array(prescribed_dofs)

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
                struct_relaxation_factor=relaxation_factor,
            )

            delta_n = vmap(hg_to_d)(
                struct_case_n.hg, struct_case_np1.hg
            )  # [n_nodes, 6]

            tot_n = vmap(hg_to_d)(struct_case_np1.hg, self.structure.hg0)

            if struct_case_n.f_ext_aero is None:
                delta_f = None
            else:
                delta_f = struct_case_n.f_ext_aero - struct_case_np1.f_ext_aero  # [n_nodes, 6]

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

        fsi_converge_status = ConvergenceStatus(self.fsi_convergence_settings)
        fsi_converge_status.print_header(dynamic=False)

        convergence_status, struct_case, aero_case = jax.lax.while_loop(
            lambda args_: ~args_[0].get_status(),
            lambda args_: _convergence_loop(*args_),  # type: ignore
            (
                fsi_converge_status,
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

        fsi_converge_status.print_footer(dynamic=False)

        return StaticAeroelastic(structure=struct_case, aero=aero_case)

    def dynamic_solve(self,
                      init_case: Optional[StaticAeroelastic | DynamicAeroelastic | DynamicAeroelasticSnapshot],
                      prescribed_dofs: Sequence[int] | Array | slice | int | None,
                      dt: Array | float,
                      n_tstep: int,
                      f_ext_follower: Optional[Array] = None,
                      f_ext_dead: Optional[Array] = None,
                      t_init: float = 0.0,
                      load_steps: int = 1,
                      struct_relaxation_factor: float = 1.0,
                      gamma_dot_relaxation_factor: float = 0.7,
                      spectral_radius: float = 0.9,
                      free_wake: bool = False,
                      include_unsteady_aero_force: bool = True,
                      ) -> DynamicAeroelastic:

        warn_if_32_bit()

        # degrees of freedom to constrain or solve for
        prescribed_dofs: Array = self.structure.make_prescribed_dofs_array(prescribed_dofs)
        solve_dofs: Array = jnp.setdiff1d(
            jnp.arange(self.structure.n_dof),
            prescribed_dofs,
            size=self.structure.n_dof - prescribed_dofs.size,
        )

        t = jnp.arange(n_tstep) * dt + t_init

        self.structure.time_integrator = TimeIntregrator(spectral_radius=spectral_radius, dt=dt)

        # initialise aeroelastic case object
        if init_case is None:
            case: DynamicAeroelastic = DynamicAeroelastic.initialise(
                initial_snapshot=self.reference_configuration(prescribed_dofs=prescribed_dofs,
                                                              use_f_ext_follower=f_ext_follower is not None,
                                                              use_f_ext_dead=f_ext_dead is not None,
                                                              use_f_ext_aero=True).to_dynamic(t=None),
                t=t, use_f_ext_follower=f_ext_follower is not None, use_f_ext_dead=f_ext_dead is not None,
                aeroelastic_object=self)

            # set forces at timestep 0
            # this is important as the time integration scheme refers to these values when solving the first timestep
            if f_ext_follower is not None:
                case.structure.f_ext_follower = case.structure.f_ext_follower.at[0, ...].set(f_ext_follower[0, ...])
            if f_ext_dead is not None:
                case.structure.f_ext_dead = case.structure.f_ext_dead.at[0, ...].set(
                    self.structure.make_f_dead_ext(f_ext=f_ext_dead[0, ...], rmat=case.structure.hg[0, :, :3, :3]))


        elif isinstance(init_case, StaticAeroelastic | DynamicAeroelasticSnapshot):
            case = DynamicAeroelastic.initialise(initial_snapshot=init_case, t=t,
                                                 use_f_ext_follower=f_ext_follower is not None,
                                                 use_f_ext_dead=f_ext_dead is not None, aeroelastic_object=self)
        elif isinstance(init_case, DynamicAeroelastic):
            if init_case.aero.n_tstep != 1: raise ValueError("init_case.aero.n_tstep != 1")
            case = DynamicAeroelastic.initialise(initial_snapshot=init_case[0], t=t,
                                                 use_f_ext_follower=f_ext_follower is not None,
                                                 use_f_ext_dead=f_ext_dead is not None, aeroelastic_object=self)
        else:
            raise NotImplementedError

        fsi_converge_status: ConvergenceStatus = ConvergenceStatus(self.fsi_convergence_settings)
        fsi_converge_status.print_header(dynamic=True)

        out = self.structure.base_dynamic_solve(struct_case=case.structure,
                                                struct_convergence_status=ConvergenceStatus(
                                                    self.structure.struct_convergence_settings),
                                                t=t,
                                                struct_relaxation_factor=struct_relaxation_factor,
                                                solve_dofs=solve_dofs,
                                                load_steps=load_steps,
                                                f_ext_follower=f_ext_follower,
                                                f_ext_dead=f_ext_dead,
                                                aero_obj=self.aero,
                                                aero_case=case.aero,
                                                fsi_convergence_status=fsi_converge_status,
                                                free_wake=free_wake,
                                                include_unsteady_aero_force=include_unsteady_aero_force,
                                                gamma_dot_relaxation_factor=gamma_dot_relaxation_factor)

        fsi_converge_status.print_footer(dynamic=True)
        return out
