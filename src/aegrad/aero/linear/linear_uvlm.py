from __future__ import annotations

from typing import Sequence, TYPE_CHECKING, Optional
from enum import Enum

from jax import Array, vmap
import jax.numpy as jnp
import os
from pathlib import Path

from aegrad.aero.data_structures import DynamicAeroCase, AeroSnapshot
from aegrad.aero.gradients.data_structures import AeroStates
from aegrad.aero.linear.data_structures import (
    AeroLinearResult,
    AeroInputUnflattened,
    AeroStateUnflattened,
    AeroOutputUnflattened,
)
from aegrad.aero.utils import (
    calculate_steady_forcing,
)
from aegrad.algebra.array_utils import ArrayList, split_to_vertex
from aegrad.aero.flowfields import FlowField
from aegrad.aero.utils import biot_savart_cutoff, KernelFunction
from aegrad.utils.linear import (
    LinearModel,
    SliceEntry,
    LinearComponent,
)
from aegrad.utils.print_utils import warn
from aegrad.aero.aic import compute_v_ind

if TYPE_CHECKING:
    from aegrad.aero.uvlm import UVLM


class LinearWakeType(Enum):
    # (is prescribed, is free)
    FROZEN = (False, False)
    PRESCRIBED = (True, False)
    FREE = (True, True)


class LinearUVLM(LinearModel):
    r"""
    Class to represent a linearised UVLM aerodynamic system about a reference state.
    """

    def __init__(
        self,
        case: UVLM,
        reference: AeroSnapshot,
        wake_type: LinearWakeType = LinearWakeType.FREE,
        bound_upwash: bool = True,
        wake_upwash: bool = True,
        unsteady_force: bool = True,
        gamma_dot_state: bool = False,
        *,
        skip_checks: bool = False,
        skip_linearisation: bool = False,
    ):
        r"""
        Initialise linear UVLM system about a reference state.
        :param case: UVLM case object to linearise.
        :param reference: StaticAero representing the reference state for linearisation.
        :param wake_type: Instance of LinearWakeType enum to specify wake treatment.
        :param bound_upwash: If true, include bound surface upwash velocities as inputs.
        :param wake_upwash: If true, include wake surface upwash velocities as inputs.
        :param unsteady_force: If true, include unsteady force.
        :param gamma_dot_state: If true, include bound circulation time derivative as a state.
        """

        # options
        self.prescribed_wake, self.free_wake = wake_type.value
        self.unsteady_force: bool = unsteady_force
        self.bound_upwash: bool = bound_upwash
        self.wake_upwash: bool = wake_upwash
        self.gamma_dot_state: bool = gamma_dot_state

        # time info
        super().__init__(reference=reference, dt=case.dt)

        # check that the reference state is steady
        # whilst linearisation can be performed about unsteady states, the current implementation omits some terms
        # required for this, however, cannot see a practical use case for such a model. Warn the user if the reference
        # state appears unsteady.
        if (
            not skip_checks
            and max([jnp.abs(zbd).max() for zbd in reference.zeta_b_dot]) > 1e-6
        ):
            warn(
                "Reference bound surface velocities are non-zero. Ensure that the reference state is steady for linearisation."
            )

        if (
            not skip_checks
            and max([jnp.abs(gbd).max() for gbd in reference.gamma_b_dot]) > 1e-6
        ):
            warn(
                "Reference bound circulation time derivative is non-zero. Ensure that the reference state is steady for linearisation."
            )

        # kernels
        self.kernels_b: Sequence[KernelFunction] = reference.n_surf * [
            biot_savart_cutoff
        ]
        self.kernels_w: Sequence[KernelFunction] = reference.n_surf * [
            biot_savart_cutoff
        ]

        # wake propagation deltas
        self.case: UVLM = case

        # linear system
        if not skip_linearisation:
            self.sys = self.linearise()

    def extract_reference_inputs(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        return {
            "zeta_b": self.reference.zeta_b,
            "zeta_b_dot": self.reference.zeta_b_dot,
            "nu_b": ArrayList.zeros_like(self.reference.zeta_b)
            if self.bound_upwash
            else None,
            "nu_w": ArrayList.zeros_like(self.reference.zeta_w)
            if self.wake_upwash
            else None,
        }

    def extract_reference_states(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        return {
            "gamma_b": self.reference.gamma_b,
            "gamma_w": self.reference.gamma_w,
            "gamma_bm1": self.reference.gamma_b if self.unsteady_force else None,
            "gamma_b_dot": self.reference.gamma_b_dot
            if self.gamma_dot_state and self.unsteady_force
            else None,
            "zeta_w": self.reference.zeta_w if self.prescribed_wake else None,
            "zeta_b": self.reference.zeta_b if self.prescribed_wake else None,
        }

    def extract_reference_outputs(
        self,
    ) -> dict[str, Optional[Array | ArrayList]]:
        return {
            "f_steady": self.reference.f_steady,
            "f_unsteady": self.reference.f_unsteady if self.unsteady_force else None,
        }

    @property
    def input_object(self) -> type[AeroInputUnflattened]:
        return AeroInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[AeroStateUnflattened]:
        return AeroStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[AeroOutputUnflattened]:
        return AeroOutputUnflattened

    def _make_input_slices(
        self, reference: AeroSnapshot
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            SliceEntry("zeta_b", True, reference.zeta_b.shape),
            SliceEntry("zeta_b_dot", True, reference.zeta_b.shape),
            SliceEntry(
                "nu_b",
                *(
                    (True, reference.zeta_b.shape)
                    if self.bound_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "nu_w",
                *(
                    (True, reference.zeta_w.shape)
                    if self.wake_upwash
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: AeroSnapshot
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry("gamma_b", True, reference.gamma_b.shape),
            SliceEntry("gamma_w", True, reference.gamma_w.shape),
            SliceEntry(
                "gamma_bm1",
                *(
                    (True, reference.gamma_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
            SliceEntry(
                "gamma_b_dot",
                *(
                    (True, reference.gamma_b.shape)
                    if self.gamma_dot_state
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_w",
                *(
                    (True, reference.zeta_w.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_b",
                *(
                    (True, reference.zeta_b.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: AeroSnapshot
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry("f_steady", True, reference.zeta_b.shape),
            SliceEntry(
                "f_unsteady",
                *(
                    (True, reference.zeta_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def f_step(self, x_vec: Array, u_vec: Array) -> tuple[Array, Array]:
        r"""
        Combined state/output step in vector form, operating on total (reference + perturbation) quantities and
        returning perturbations relative to the reference.
        :param x_vec: State vector, [n_states]
        :param u_vec: Input vector, [n_inputs]
        :return: Tuple of (state perturbation vector, output perturbation vector).
        """
        ref = self.reference
        u = self._unpack_input_vector(u_vec)
        x = self._unpack_state_vector(x_vec)

        assert isinstance(x, AeroStateUnflattened) and isinstance(
            u, AeroInputUnflattened
        ), (
            "Unpacked state and input must be of type AeroStateUnflattened and AeroInputUnflattened."
        )

        # total quantities
        zeta_b_tot = ArrayList([zr + du for zr, du in zip(ref.zeta_b, u.zeta_b)])
        zeta_b_dot_tot = ArrayList(
            [zr + du for zr, du in zip(ref.zeta_b_dot, u.zeta_b_dot)]
        )

        gamma_b_tot = ArrayList([gr + dg for gr, dg in zip(ref.gamma_b, x.gamma_b)])
        gamma_w_tot = ArrayList([gr + dg for gr, dg in zip(ref.gamma_w, x.gamma_w)])
        gamma_b_dot_tot = (
            ArrayList([gr + dg for gr, dg in zip(ref.gamma_b_dot, x.gamma_b_dot)])
            if x.gamma_b_dot is not None
            else ref.gamma_b_dot
        )
        zeta_w_tot = (
            ArrayList([zr + dz for zr, dz in zip(ref.zeta_w, x.zeta_w)])
            if x.zeta_w is not None
            else ref.zeta_w
        )

        q_nm1 = AeroStates(
            gamma_b=gamma_b_tot,
            gamma_w=gamma_w_tot,
            gamma_b_dot=gamma_b_dot_tot,
            zeta_w=zeta_w_tot,
        )
        (
            _,
            _,
            gamma_b_np1,
            gamma_w_np1,
            gamma_b_dot_np1,
            _,
            zeta_w_np1,
            _,
            _,
            _,
        ) = self.case.base_solve_from_grid(
            q_nm1=q_nm1,
            t_n=ref.t,
            zeta_b_n=zeta_b_tot,
            zeta_b_nm1=ref.zeta_b,
            zeta_b_dot_n=zeta_b_dot_tot,
            static=False,
            horseshoe=False,
            linearise_variable_wake=True,
            nu_b=u.nu_b if self.bound_upwash else None,
            nu_w=u.nu_w if (self.prescribed_wake and self.wake_upwash) else None,
        )

        rho = ref.flowfield.rho

        def v_out_func(x_target: Array) -> Array:
            return ref.flowfield.vmap_call(x=x_target, t=ref.t) + compute_v_ind(
                cs=x_target,
                zetas=ArrayList([*zeta_b_tot, *zeta_w_tot]),
                gammas=ArrayList([*gamma_b_tot, *gamma_w_tot]),
                kernels=[*self.kernels_b, *self.kernels_w],
                batch_size=self.case.batch_size,
                mirror_normal=self.case.mirror_normal,
                mirror_point=self.case.mirror_point,
            )

        f_steady_out = calculate_steady_forcing(
            zeta_b=zeta_b_tot,
            zeta_dot_b=zeta_b_dot_tot,
            gamma_b=gamma_b_tot,
            gamma_w=gamma_w_tot,
            rho=rho,
            v_func=v_out_func,
            v_inputs=u.nu_b if self.bound_upwash else None,
        )

        if self.unsteady_force:
            if self.gamma_dot_state:
                assert x.gamma_b_dot is not None
                gamma_b_dot_at_n = ArrayList(
                    [gr + dg for gr, dg in zip(ref.gamma_b_dot, x.gamma_b_dot)]
                )
            else:
                assert x.gamma_bm1 is not None
                gamma_b_dot_at_n = ArrayList(
                    [(gn - gm1) / self.dt for gn, gm1 in zip(x.gamma_b, x.gamma_bm1)]
                )
            f_unsteady_out = ArrayList(
                [
                    split_to_vertex(
                        rho * gamma_b_dot_at_n[i][..., None] * ref.nc[i], (0, 1)
                    )
                    for i in range(ref.n_surf)
                ]
            )
        else:
            f_unsteady_out = None

        # pack perturbations
        d_gamma_b = ArrayList([gn - gr for gn, gr in zip(gamma_b_np1, ref.gamma_b)])
        d_gamma_w = ArrayList([gn - gr for gn, gr in zip(gamma_w_np1, ref.gamma_w)])
        d_zeta_w = (
            ArrayList([zn - zr for zn, zr in zip(zeta_w_np1, ref.zeta_w)])
            if self.prescribed_wake
            else None
        )

        assert gamma_b_dot_np1 is not None
        d_gamma_bm1 = x.gamma_b if self.unsteady_force else None
        d_gamma_b_dot = (
            ArrayList([gn - gr for gn, gr in zip(gamma_b_dot_np1, ref.gamma_b_dot)])
            if self.gamma_dot_state
            else None
        )
        d_zeta_b_state = u.zeta_b if self.prescribed_wake else None

        d_f_steady = ArrayList([fn - fr for fn, fr in zip(f_steady_out, ref.f_steady)])
        if self.unsteady_force:
            assert f_unsteady_out is not None and ref.f_unsteady is not None
            d_f_unsteady = ArrayList(
                [fn - fr for fn, fr in zip(f_unsteady_out, ref.f_unsteady)]
            )
        else:
            d_f_unsteady = None

        state_new = AeroStateUnflattened(
            gamma_b=d_gamma_b,
            gamma_w=d_gamma_w,
            gamma_bm1=d_gamma_bm1,
            gamma_b_dot=d_gamma_b_dot,
            zeta_w=d_zeta_w,
            zeta_b=d_zeta_b_state,
        )
        out_new = AeroOutputUnflattened(f_steady=d_f_steady, f_unsteady=d_f_unsteady)

        return self._pack_state_vector(state_new), self._pack_output_vector(out_new)

    def run(
        self,
        u: AeroInputUnflattened,
        x0: Optional[AeroStateUnflattened] = None,
        flowfield: Optional[FlowField] = None,
        use_matrix=False,
    ) -> AeroLinearResult:
        r"""
        Run the linear system.
        :param u: Total input over time (reference + pertubation).
        :param x0: Initial state perturbations, defaults to zero state.
        :param flowfield: FlowField object to provide flow velocities for bound and wake upwash, defaults to no flow.
        :param use_matrix: If true, use explicit matrix representation for linear system, otherwise use operator form.
        """
        if self.prescribed_wake and self.sys.removed_u_np1:
            warn(
                "Wake perturbations coordinates at the trailing edge are zero when removing u_np1 from the system."
            )

        if x0 is None:
            x0_vec = None
        else:
            x0_vec = self._pack_state_vector_t(x0)

        n_tstep: int = u.zeta_b[0].shape[
            0
        ]  # number of time steps from first surface, first entry
        t = self.reference.t + jnp.arange(0, n_tstep) * self.dt  # time vector

        u_tot = u

        if self.bound_upwash and flowfield is None and u_tot.nu_b is None:
            warn(
                "No flowfield or bound upwash perturbations provided. Assuming zero bound upwash perturbations."
            )
            u_tot.nu_b = ArrayList(
                [jnp.zeros((n_tstep, *zb.shape)) for zb in self.reference.zeta_b]
            )

        if self.wake_upwash and flowfield is None and u_tot.nu_w is None:
            warn(
                "No flowfield or wake upwash perturbations provided. Assuming zero wake upwash perturbations."
            )
            u_tot.nu_w = ArrayList(
                [jnp.zeros((n_tstep, *zw.shape)) for zw in self.reference.zeta_w]
            )

        # add flowfield contributions to input upwash if provided
        if flowfield is not None:
            if self.bound_upwash:
                nu_b_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_b_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_b[i_surf],
                            t,  # type: ignore
                        )
                        - flowfield.vmap_call(self.reference.zeta_b[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_b is None:
                    u_tot.nu_b = nu_b_flow
                else:
                    u_tot.nu_b += nu_b_flow
            if self.wake_upwash:
                nu_w_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_w_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_w[i_surf],
                            t,  # type: ignore
                        )
                        - flowfield.vmap_call(self.reference.zeta_w[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_w is None:
                    u_tot.nu_w = nu_w_flow
                else:
                    u_tot.nu_w += nu_w_flow
        u_vec = self._pack_input_vector_t(u_tot)

        # run linear system
        x_t, y_t = self.sys.run(u_vec, x0_vec, use_matrix=use_matrix)

        x_t_obj = self._unpack_state_vector_t(x_t)
        y_t_obj = self._unpack_output_vector_t(y_t)

        assert isinstance(x_t_obj, AeroStateUnflattened) and isinstance(
            y_t_obj, AeroOutputUnflattened
        ), (
            "Unpacked state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        x_t_tot_obj = self.get_total_state_t(x_t_obj)
        y_t_tot_obj = self.get_total_output_t(y_t_obj)
        u_t_tot_obj = self.get_total_input_t(u_tot)

        assert (
            isinstance(u_t_tot_obj, AeroInputUnflattened)
            and isinstance(x_t_tot_obj, AeroStateUnflattened)
            and isinstance(y_t_tot_obj, AeroOutputUnflattened)
        ), (
            "Unpacked total state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        assert isinstance(self.reference, AeroSnapshot), (
            "Reference state must be of type AeroSnapshot."
        )

        # save results to object
        return AeroLinearResult(
            reference=self.reference,
            u_t=u,
            x_t=x_t_obj,
            y_t=y_t_obj,
            u_t_tot=u_t_tot_obj,
            x_t_tot=x_t_tot_obj,
            y_t_tot=y_t_tot_obj,
            n_tstep=n_tstep,
            t=t,
            n_surf=self.reference.n_surf,
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
        )

    def reference_snapshot(self) -> DynamicAeroCase:
        r"""
        Get the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
        :return: StaticAero at reference state
        """
        return DynamicAeroCase(
            zeta_b=self.reference.zeta_b,
            zeta_b_dot=self.reference.zeta_b_dot,
            zeta_w=self.reference.zeta_w,
            gamma_b=self.reference.gamma_b,
            gamma_b_dot=self.reference.gamma_b_dot,
            gamma_w=self.reference.gamma_w,
            f_steady=self.reference.f_steady,
            f_unsteady=self.reference.f_unsteady,
            cs_ang=self.reference.cs_ang,
            cs_vel=self.reference.cs_vel,
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
            i_ts=jnp.atleast_1d(-1),
            t=jnp.zeros(()),
            c=self.reference.c,
            n=self.reference.nc,
            kernels=self.reference.kernels,
            mirror_normal=self.reference.mirror_normal,
            mirror_point=self.reference.mirror_point,
            flowfield=self.reference.flowfield,
            dof_mapping=self.reference.dof_mapping,
            free_wake=self.reference.free_wake,
            gamma_dot_relaxation=self.reference.gamma_dot_relaxation,
            static_horseshoe=self.reference.static_horseshoe,
            batch_size=self.case.batch_size,
        )

    def plot_reference(
        self, directory: os.PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), index=None, plot_wake=plot_wake
        )
