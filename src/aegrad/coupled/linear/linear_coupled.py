from __future__ import annotations
from typing import Literal, Optional, TYPE_CHECKING

from jax import numpy as jnp, Array, vmap

from aegrad.aero.linear import LinearWakeType, LinearUVLM
from aegrad.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroStateUnflattened,
    AeroOutputUnflattened,
)
from aegrad.aero.utils import project_forcing_to_beam
from aegrad.algebra.array_utils import ArrayList
from aegrad.algebra.se3 import exp_se3, ha_to_ha_tilde
from aegrad.coupled import StaticAeroelastic
from aegrad.coupled.linear.data_structures import (
    AeroelasticInputUnflattened,
    AeroelasticStateUnflattened,
    AeroelasticOutputUnflattened,
    AeroelasticLinearResult,
)
from aegrad.structure.utils import get_solve_dofs, transform_nodal_vect
from aegrad.structure.linear.linear_beam import LinearBeam
from aegrad.utils.constants import BASE_LOBATTO_ORDER
from aegrad.utils.linear import LinearComponent, SliceEntry, LinearModel

if TYPE_CHECKING:
    from aegrad.coupled.coupled import BaseCoupledAeroelastic


class LinearCoupled(LinearModel):
    def __init__(
        self,
        case: BaseCoupledAeroelastic,
        reference: StaticAeroelastic,
        wake_type: LinearWakeType = LinearWakeType.FREE,
        bound_upwash: bool = True,
        wake_upwash: bool = False,
        unsteady_force: bool = True,
        gamma_dot_state: bool = False,
        local_forcing: bool = False,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        *,
        skip_checks: bool = False,
    ):
        self.reference: StaticAeroelastic = reference

        self._aero = LinearUVLM(
            case=case.aero,
            reference=reference.aero,
            wake_type=wake_type,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
            gamma_dot_state=gamma_dot_state,
            skip_linearisation=True,
            skip_checks=skip_checks,
        )
        self._beam = LinearBeam(
            beam=case.structure,
            reference=reference.structure,
            dt=case.aero.dt,
            local_forcing=local_forcing,
            int_order=int_order,
        )

        self.n_beam_dof: int = case.structure.n_dof - len(
            reference.structure.prescribed_dofs
        )
        self.n_nodes: int = case.structure.n_nodes
        self.free_dofs: Array = jnp.array(
            get_solve_dofs(
                n_dof=case.structure.n_dof,
                prescribed_dofs=reference.structure.prescribed_dofs,
            )
        )

        super().__init__(reference=reference, dt=case.aero.dt)

        self.sys = self.linearise()

    def extract_reference_inputs(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        inputs = (
            self._aero.extract_reference_inputs()
            | self._beam.extract_reference_inputs()
        )

        # remove redundant
        inputs.pop("zeta_b")
        inputs.pop("zeta_b_dot")
        return inputs

    def extract_reference_states(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        states = (
            self._aero.extract_reference_states()
            | self._beam.extract_reference_states()
        )

        # remove redundant
        states.pop("zeta_b")
        return states

    def extract_reference_outputs(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        states = (
            self._aero.extract_reference_outputs()
            | self._beam.extract_reference_outputs()
        )

        # remove redundant
        states.pop("f_steady")
        states.pop("f_unsteady")
        return states

    @property
    def input_object(self) -> type[AeroelasticInputUnflattened]:
        return AeroelasticInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[AeroelasticStateUnflattened]:
        return AeroelasticStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[AeroelasticOutputUnflattened]:
        return AeroelasticOutputUnflattened

    def _make_input_slices(
        self, reference: StaticAeroelastic
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            SliceEntry(
                "nu_b",
                *(
                    (True, reference.aero.zeta_b.shape)
                    if self._aero.bound_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "nu_w",
                *(
                    (True, reference.aero.zeta_w.shape)
                    if self._aero.wake_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "f_ext_follower",
                True,
                (self.n_beam_dof,),
            ),
            SliceEntry("f_ext_dead", True, (self.n_beam_dof,)),
        )
        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: StaticAeroelastic
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry("gamma_b", True, reference.aero.gamma_b.shape),
            SliceEntry("gamma_w", True, reference.aero.gamma_w.shape),
            SliceEntry(
                "gamma_bm1",
                *(
                    (True, reference.aero.gamma_b.shape)
                    if self._aero.unsteady_force
                    else (False, None)
                ),
            ),
            SliceEntry(
                "gamma_b_dot",
                *(
                    (True, reference.aero.gamma_b.shape)
                    if self._aero.gamma_dot_state
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_w",
                *(
                    (True, reference.aero.zeta_w.shape)
                    if self._aero.prescribed_wake
                    else (False, None)
                ),
            ),
            SliceEntry("q", True, (self.n_beam_dof,)),
            SliceEntry("q_dot", True, (self.n_beam_dof,)),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: StaticAeroelastic
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry("q", True, (self.n_beam_dof,)),
            SliceEntry("q_dot", True, (self.n_beam_dof,)),
        )
        return self._make_slices(slice_entries)

    def f_step(self, x_vec: Array, u_vec: Array) -> tuple[Array, Array]:
        ref = self.reference
        u = self._unpack_input_vector(u_vec)
        x = self._unpack_state_vector(x_vec)

        assert isinstance(x, AeroelasticStateUnflattened) and isinstance(
            u, AeroelasticInputUnflattened
        ), (
            "Unpacked state and input must be of type AeroelasticStateUnflattened and AeroelasticInputUnflattened."
        )

        # fill in prescribed dofs with zeros
        delta_q_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(x.q)
            .reshape(self.n_nodes, 6)
        )
        delta_q_dot_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(x.q_dot)
            .reshape(self.n_nodes, 6)
        )

        # total perturbed coordinates and time derivative
        hg = jnp.einsum("ijk,ikl->ijl", ref.structure.hg, vmap(exp_se3)(delta_q_full))
        hg_dot = jnp.einsum(
            "ijk,ikl->ijl",
            ref.structure.hg,
            vmap(ha_to_ha_tilde)(delta_q_dot_full),
        )

        # aerodynamic grid
        zeta_b = self._aero.case.hg_to_zeta_b(hg_n=hg, cs_ang_n=self._aero.case.cs_ang0)
        zeta_b_dot = self._aero.case.hg_dot_to_zeta_b_dot(
            hg_n=hg,
            hg_dot_n=hg_dot,
            cs_ang_n=self._aero.case.cs_ang0,
            cs_vel_n=self._aero.case.cs_vel0,
        )

        # aero step consumes perturbations of the aero grid, not totals
        d_zeta_b = ArrayList([zt - zr for zt, zr in zip(zeta_b, ref.aero.zeta_b)])
        d_zeta_b_dot = ArrayList(
            [zt - zr for zt, zr in zip(zeta_b_dot, ref.aero.zeta_b_dot)]
        )

        # pass through aerodynamic system
        u_aero_in = AeroInputUnflattened(
            zeta_b=d_zeta_b, zeta_b_dot=d_zeta_b_dot, nu_b=u.nu_b, nu_w=u.nu_w
        )
        x_aero_in = AeroStateUnflattened(
            gamma_b=x.gamma_b,
            gamma_w=x.gamma_w,
            gamma_b_dot=x.gamma_b_dot,
            gamma_bm1=x.gamma_bm1,
            zeta_w=x.zeta_w,
            zeta_b=d_zeta_b if self._aero.prescribed_wake else None,
        )

        u_aero_in_vec = self._aero._pack_input_vector(u_aero_in)
        x_aero_in_vec = self._aero._pack_state_vector(x_aero_in)

        x_aero_out_vec, y_aero_out_vec = self._aero.f_step(
            x_vec=x_aero_in_vec, u_vec=u_aero_in_vec
        )

        x_aero_out = self._aero._unpack_state_vector(x=x_aero_out_vec)
        y_aero_out = self._aero._unpack_output_vector(y=y_aero_out_vec)
        assert isinstance(x_aero_out, AeroStateUnflattened) and isinstance(
            y_aero_out, AeroOutputUnflattened
        ), (
            "Unpacked aero state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        # total aero forces on the grid, combining perturbation and reference
        f_aero = ArrayList(
            [fp + fr for fp, fr in zip(y_aero_out.f_steady, ref.aero.f_steady)]
        )
        if self._aero.unsteady_force:
            assert y_aero_out.f_unsteady is not None and ref.aero.f_unsteady is not None
            f_aero = ArrayList(
                [
                    ft + fp + fr
                    for ft, fp, fr in zip(
                        f_aero, y_aero_out.f_unsteady, ref.aero.f_unsteady
                    )
                ]
            )

        # project total aero forces onto the beam (global frame)
        rmat = hg[:, :3, :3]
        f_aero_beam_global = project_forcing_to_beam(
            f_total=f_aero,
            rmat=rmat,
            dof_mapping=self._aero.case.dof_mapping,
            x0_aero=self._aero.case.x0_b,
        )

        # convert to the frame expected by the beam's linear system
        if self._beam.local_forcing:
            f_aero_beam = transform_nodal_vect(
                f_aero_beam_global, jnp.swapaxes(rmat, -1, -2)
            )
        else:
            f_aero_beam = f_aero_beam_global

        # reference aero-projected force, in beam-consistent frame
        ref_f_aero = ref.structure.f_ext_aero
        if ref_f_aero is None:
            ref_f_aero = jnp.zeros((self.n_nodes, 6))
        elif not self._beam.local_forcing:
            ref_f_aero = transform_nodal_vect(ref_f_aero, ref.structure.hg[:, :3, :3])

        d_f_aero_free = (f_aero_beam - ref_f_aero).ravel()[self.free_dofs]

        # user-provided perturbations are supplied directly at free DoFs
        assert u.f_ext_follower is not None and u.f_ext_dead is not None
        d_f_ext_free = u.f_ext_follower + u.f_ext_dead

        # beam step in perturbation form (discrete-time Tustin)
        x_beam = jnp.concatenate([x.q, x.q_dot])
        x_beam_np1 = self._beam.sys.a @ x_beam + self._beam.sys.b @ (
            d_f_aero_free + d_f_ext_free
        )
        d_q_np1 = x_beam_np1[: self.n_beam_dof]
        d_q_dot_np1 = x_beam_np1[self.n_beam_dof :]

        state_np1 = AeroelasticStateUnflattened(
            gamma_b=x_aero_out.gamma_b,
            gamma_w=x_aero_out.gamma_w,
            gamma_bm1=x_aero_out.gamma_bm1,
            gamma_b_dot=x_aero_out.gamma_b_dot,
            zeta_w=x_aero_out.zeta_w,
            q=d_q_np1,
            q_dot=d_q_dot_np1,
        )
        output_n = AeroelasticOutputUnflattened(q=x.q, q_dot=x.q_dot)
        return self._pack_state_vector(state_np1), self._pack_output_vector(output_n)

    def run(
        self,
        u: AeroelasticInputUnflattened,
        x0: Optional[AeroelasticStateUnflattened] = None,
        use_matrix=False,
    ) -> AeroelasticLinearResult: ...
