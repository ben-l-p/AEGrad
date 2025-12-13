from __future__ import annotations
from jax import Array, jit, vmap
import jax
import jax.numpy as jnp
from typing import Sequence, TYPE_CHECKING, Optional, Self
from functools import reduce
from operator import mul
from enum import Enum
from os import PathLike
from pathlib import Path

from aegrad.aero.data_structures import (
    AeroSnapshot,
    InputSlices,
    StateSlices,
    OutputSlices,
    LinearComponent,
    _SliceEntry,
    InputUnflattened,
    StateUnflattened,
    OutputUnflattened,
)
from aegrad.aero.uvlm_utils import get_c, get_nc, propagate_wake, steady_forcing
from aegrad.algebra.linear_operators import LinearOperator, LinearSystem
from aegrad.algebra.array_utils import ArrayList, split_to_vertex
from aegrad.aero.aic import compute_aic_sys_assembled
from aegrad.aero.flowfields import FlowField
from aegrad.aero.kernels import KernelFunction, biot_savart_cutoff
from aegrad.utils import shallow_asdict, replace_self
from aegrad.print_output import print_with_time, warn
from aegrad.plotting.pvd import write_pvd

if TYPE_CHECKING:
    from aegrad.aero.case import AeroCase


class LinearWakeType(Enum):
    # (is prescribed, is free)
    FROZEN = (False, False)
    PRESCRIBED = (True, False)
    FREE = (True, True)


class LinearAero:
    def __init__(
        self,
        case: AeroCase,
        reference: AeroSnapshot,
        wake_type: LinearWakeType = LinearWakeType.FREE,
        bound_upwash: bool = True,
        wake_upwash: bool = True,
        unsteady_force: bool = True,
        gamma_dot_state: bool = False,
    ):
        # options
        self.prescribed_wake, self.free_wake = wake_type.value
        self.unsteady_force: bool = unsteady_force
        self.bound_upwash: bool = bound_upwash
        self.wake_upwash: bool = wake_upwash
        self.gamma_dot_state: bool = gamma_dot_state

        # save names from case
        self.surf_b_names: list[str] = [f"linear_{name}" for name in case.surf_b_names]
        self.surf_w_names: list[str] = [f"linear_{name}" for name in case.surf_w_names]

        # time info
        self.dt: Array = case.dt
        self.t0: Array = reference.t

        if max([jnp.abs(zbd).max() for zbd in reference.zeta_b_dot]) > 1e-6:
            warn(
                "Reference bound surface velocities are non-zero. Ensure that the reference state is steady for linearisation."
            )

        if max([jnp.abs(gbd).max() for gbd in reference.gamma_b_dot]) > 1e-6:
            warn(
                "Reference bound circulation time derivative is non-zero. Ensure that the reference state is steady for linearisation."
            )

        # reference state
        self.n_surf: int = reference.n_surf
        self.zeta0_b: ArrayList = ArrayList(reference.zeta_b)
        self.zeta0_b_dot: ArrayList = ArrayList(reference.zeta_b_dot)
        self.zeta0_w: ArrayList = ArrayList(reference.zeta_w)
        self.gamma0_b: ArrayList = ArrayList(reference.gamma_b)
        self.gamma0_b_dot: ArrayList = ArrayList(reference.gamma_b_dot)
        self.gamma0_w: ArrayList = ArrayList(reference.gamma_w)
        self.f_steady0: ArrayList = ArrayList(reference.f_steady)
        self.f_unsteady0: ArrayList = ArrayList(reference.f_unsteady)
        self.c0: ArrayList = get_c(reference.zeta_b)
        self.n0: ArrayList = get_nc(reference.zeta_b)
        self.flowfield0: FlowField = case.flowfield

        # baseline shapes
        self.zeta_b_shapes: Sequence[tuple[int, ...]] = [
            arr.shape for arr in self.zeta0_b
        ]
        self.zeta_w_shapes: Sequence[tuple[int, ...]] = [
            arr.shape for arr in self.zeta0_w
        ]
        self.gamma_b_shapes: Sequence[tuple[int, ...]] = [
            arr.shape for arr in self.gamma0_b
        ]
        self.gamma_w_shapes: Sequence[tuple[int, ...]] = [
            arr.shape for arr in self.gamma0_w
        ]

        # slices of individial surface components in full vector
        self.input_slices, self.n_inputs = self._make_input_slices()
        self.state_slices, self.n_states = self._make_state_slices()
        self.output_slices, self.n_outputs = self._make_output_slices()

        # kernels
        self.kernels_b: Sequence[KernelFunction] = self.n_surf * [biot_savart_cutoff]
        self.kernels_w: Sequence[KernelFunction] = self.n_surf * [biot_savart_cutoff]

        # wake propagation deltas
        self.delta_w: Sequence[Optional[Array]] = case.delta_w

        # linear operators for system
        self.base_sys: LinearSystem = self.linearise()

        # final system - this is overwritten for updating models
        self.sys: LinearSystem = self.base_sys

        # system results, if simulated
        self._u_t: Optional[InputUnflattened] = None
        self._x_t: Optional[StateUnflattened] = None
        self._y_t: Optional[OutputUnflattened] = None
        self._u_t_tot: Optional[InputUnflattened] = None
        self._x_t_tot: Optional[StateUnflattened] = None
        self._y_t_tot: Optional[OutputUnflattened] = None
        self._n_tstep: Optional[int] = None
        self._t: Optional[Array] = None

    @property
    def u_t(self):
        if self._u_t is None:
            raise ValueError("No input_ results available. Run a linear system first.")
        return self._u_t

    @property
    def x_t(self):
        if self._x_t is None:
            raise ValueError("No state results available. Run a linear system first.")
        return self._x_t

    @property
    def y_t(self):
        if self._y_t is None:
            raise ValueError("No output results available. Run a linear system first.")
        return self._y_t

    @property
    def u_t_tot(self):
        if self._u_t_tot is None:
            raise ValueError(
                "No total input_ results available. Run a linear system first."
            )
        return self._u_t_tot

    @property
    def x_t_tot(self):
        if self._x_t_tot is None:
            raise ValueError(
                "No total state results available. Run a linear system first."
            )
        return self._x_t_tot

    @property
    def y_t_tot(self):
        if self._y_t_tot is None:
            raise ValueError(
                "No total output results available. Run a linear system first."
            )
        return self._y_t_tot

    @property
    def n_tstep_tot(self) -> int:
        if self._n_tstep is None:
            raise ValueError("No solution available. Run a linear system first.")
        return self._n_tstep

    @property
    def t(self) -> Array:
        if self._t is None:
            raise ValueError("No time available. Run a linear system first.")
        return self._t

    def get_reference_inputs(self) -> InputUnflattened:
        return InputUnflattened(
            self.zeta0_b,
            self.zeta0_b_dot,
            [jnp.zeros_like(arr) for arr in self.zeta0_b]
            if self.bound_upwash
            else None,
            [jnp.zeros_like(arr) for arr in self.zeta0_w] if self.wake_upwash else None,
        )

    def get_reference_states(self) -> StateUnflattened:
        return StateUnflattened(
            self.gamma0_b,
            self.gamma0_w,
            self.gamma0_b if self.unsteady_force else None,
            self.gamma0_b_dot if self.gamma_dot_state else None,
            self.zeta0_w if self.prescribed_wake else None,
            self.zeta0_b if self.prescribed_wake else None,
        )

    def get_reference_outputs(self) -> OutputUnflattened:
        return OutputUnflattened(
            self.f_steady0, self.f_unsteady0 if self.unsteady_force else None
        )

    @staticmethod
    def _make_slices[T](
        slice_entries: Sequence[_SliceEntry], cls: type[T]
    ) -> tuple[T, int]:
        r"""
        Helper function to create slices classes for the vectors, and count the number of elements.
        Blocks should be passed in the order they are in the dataclass.
        """
        # make slices
        cnt = 0
        out_dict = {}
        for entry in slice_entries:
            if not entry.enabled:  # if disabled
                out_dict[entry.name] = LinearComponent(False, None, None)
            else:
                slices = []
                for size in [reduce(mul, shape) for shape in entry.shapes]:
                    slices.append(slice(cnt, cnt + size))
                    cnt += size
                out_dict[entry.name] = LinearComponent(True, slices, entry.shapes)
        return cls(**out_dict), cnt

    def _make_input_slices(self) -> tuple[InputSlices, int]:
        slice_entries = (
            _SliceEntry("zeta_b", True, self.zeta_b_shapes),
            _SliceEntry("zeta_b_dot", True, self.zeta_b_shapes),
            _SliceEntry(
                "nu_b",
                *((True, self.zeta_b_shapes) if self.bound_upwash else (False, None)),
            ),
            _SliceEntry(
                "nu_w",
                *((True, self.zeta_w_shapes) if self.wake_upwash else (False, None)),
            ),
        )
        return self._make_slices(slice_entries, InputSlices)

    def _make_state_slices(self) -> tuple[StateSlices, int]:
        slice_entries = (
            _SliceEntry("gamma_b", True, self.gamma_b_shapes),
            _SliceEntry("gamma_w", True, self.gamma_w_shapes),
            _SliceEntry(
                "gamma_bm1",
                *(
                    (True, self.gamma_b_shapes)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "gamma_b_dot",
                *(
                    (True, self.gamma_b_shapes)
                    if self.gamma_dot_state
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "zeta_w",
                *(
                    (True, self.zeta_w_shapes)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
            _SliceEntry(
                "zeta_b",
                *(
                    (True, self.zeta_b_shapes)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries, StateSlices)

    def _make_output_slices(self) -> tuple[OutputSlices, int]:
        slice_entries = (
            _SliceEntry("f_steady", True, self.zeta_b_shapes),
            _SliceEntry(
                "f_unsteady",
                *((True, self.zeta_b_shapes) if self.unsteady_force else (False, None)),
            ),
        )
        return self._make_slices(slice_entries, OutputSlices)

    def _unpack_vector(
        self, x: Array, slices: dict[str, LinearComponent], add_t: bool = False
    ) -> dict[str, Optional[ArrayList]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                if add_t:
                    n_tstep = x.shape[0]
                    out[name] = ArrayList(
                        [
                            x[:, entry.slices[i_surf]].reshape(
                                n_tstep, *entry.shapes[i_surf]
                            )
                            for i_surf in range(self.n_surf)
                        ]
                    )
                else:
                    out[name] = ArrayList(
                        [
                            x[entry.slices[i_surf]].reshape(entry.shapes[i_surf])
                            for i_surf in range(self.n_surf)
                        ]
                    )
        return out

    def _unpack_input_vector(self, u: Array) -> InputUnflattened:
        return InputUnflattened(
            **self._unpack_vector(u, shallow_asdict(self.input_slices))
        )

    def _unpack_state_vector(self, x: Array) -> StateUnflattened:
        return StateUnflattened(
            **self._unpack_vector(x, shallow_asdict(self.state_slices))
        )

    def _unpack_output_vector(self, y: Array) -> OutputUnflattened:
        return OutputUnflattened(
            **self._unpack_vector(y, shallow_asdict(self.output_slices))
        )

    def _unpack_input_vector_t(self, u_t: Array) -> InputUnflattened:
        return InputUnflattened(
            **self._unpack_vector(u_t, shallow_asdict(self.input_slices), add_t=True)
        )

    def _unpack_state_vector_t(self, x_t: Array) -> StateUnflattened:
        return StateUnflattened(
            **self._unpack_vector(x_t, shallow_asdict(self.state_slices), add_t=True)
        )

    def _unpack_output_vector_t(self, y_t: Array) -> OutputUnflattened:
        return OutputUnflattened(
            **self._unpack_vector(y_t, shallow_asdict(self.output_slices), add_t=True)
        )

    def _pack_vector(
        self,
        slices: dict[str, LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[ArrayList]],
    ) -> Array:
        vec = jnp.zeros(vec_length)
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.n_surf):
                    vec = vec.at[entry.slices[i_surf]].set(arrs[name][i_surf].ravel())
        return vec

    def _pack_input_vector(self, u_input: InputUnflattened) -> Array:
        return self._pack_vector(
            shallow_asdict(self.input_slices), self.n_inputs, shallow_asdict(u_input)
        )

    def _pack_state_vector(self, x_state: StateUnflattened) -> Array:
        return self._pack_vector(
            shallow_asdict(self.state_slices), self.n_states, shallow_asdict(x_state)
        )

    def _pack_output_vector(self, y_output: OutputUnflattened) -> Array:
        return self._pack_vector(
            shallow_asdict(self.output_slices), self.n_outputs, shallow_asdict(y_output)
        )

    def _pack_vector_t(
        self,
        slices: dict[str, LinearComponent],
        vec_length: int,
        arrs: dict[str, Optional[ArrayList]],
    ) -> Array:
        n_tstep = list(arrs.values())[0][0].shape[
            0
        ]  # find number of timesteps from first surface, first entry
        vec_t = jnp.zeros((n_tstep, vec_length))
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.n_surf):
                    vec_t = vec_t.at[:, entry.slices[i_surf]].set(
                        arrs[name][i_surf].reshape(n_tstep, -1)
                    )
        return vec_t

    def _pack_input_vector_t(self, u_input: InputUnflattened) -> Array:
        return self._pack_vector_t(
            shallow_asdict(self.input_slices), self.n_inputs, shallow_asdict(u_input)
        )

    def _pack_state_vector_t(self, x_state: StateUnflattened) -> Array:
        return self._pack_vector_t(
            shallow_asdict(self.state_slices), self.n_states, shallow_asdict(x_state)
        )

    def _pack_output_vector_t(self, y_output: OutputUnflattened) -> Array:
        return self._pack_vector_t(
            shallow_asdict(self.output_slices), self.n_outputs, shallow_asdict(y_output)
        )

    def _get_total(
        self,
        input_: dict[str, Optional[Sequence[Array]]],
        reference: dict[str, Optional[Sequence[Array]]],
        add_t: bool = False,
    ) -> dict[str, Optional[Sequence[Array]]]:
        out = {}
        for name, entry in reference.items():
            if entry is None:
                out[name] = None
            else:
                arrs = ArrayList([])
                for i_surf in range(self.n_surf):
                    if add_t:
                        arrs.append(
                            reference[name][i_surf][None, ...] + input_[name][i_surf]
                        )
                    else:
                        arrs.append(reference[name][i_surf] + input_[name][i_surf])
                out[name] = arrs
        return out

    def get_total_input(self, u: InputUnflattened) -> InputUnflattened:
        return InputUnflattened(
            **self._get_total(
                shallow_asdict(u), shallow_asdict(self.get_reference_inputs())
            )
        )

    def get_total_state(self, x: StateUnflattened) -> StateUnflattened:
        return StateUnflattened(
            **self._get_total(
                shallow_asdict(x), shallow_asdict(self.get_reference_states())
            )
        )

    def get_total_output(self, y: OutputUnflattened) -> OutputUnflattened:
        return OutputUnflattened(
            **self._get_total(
                shallow_asdict(y), shallow_asdict(self.get_reference_outputs())
            )
        )

    def get_total_input_t(self, u_t: InputUnflattened) -> InputUnflattened:
        return InputUnflattened(
            **self._get_total(
                shallow_asdict(u_t),
                shallow_asdict(self.get_reference_inputs()),
                add_t=True,
            )
        )

    def get_total_state_t(self, x_t: StateUnflattened) -> StateUnflattened:
        return StateUnflattened(
            **self._get_total(
                shallow_asdict(x_t),
                shallow_asdict(self.get_reference_states()),
                add_t=True,
            )
        )

    def get_total_output_t(self, y_t: OutputUnflattened) -> OutputUnflattened:
        return OutputUnflattened(
            **self._get_total(
                shallow_asdict(y_t),
                shallow_asdict(self.get_reference_outputs()),
                add_t=True,
            )
        )

    def _get_zero(
        self, slices: dict[str, LinearComponent]
    ) -> dict[str, Optional[ArrayList]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                arrs = ArrayList([])
                for i_surf in range(self.n_surf):
                    arrs.append(jnp.zeros(entry.shapes[i_surf]))
                out[name] = arrs
        return out

    def get_zero_input(self) -> InputUnflattened:
        return InputUnflattened(**self._get_zero(shallow_asdict(self.input_slices)))

    def get_zero_state(self) -> StateUnflattened:
        return StateUnflattened(**self._get_zero(shallow_asdict(self.state_slices)))

    def get_zero_output(self) -> OutputUnflattened:
        return OutputUnflattened(**self._get_zero(shallow_asdict(self.output_slices)))

    def _unflatten_subvec(self, vec: Array, component: LinearComponent) -> ArrayList:
        arrs = ArrayList([])
        cnt = 0
        for i_surf in range(self.n_surf):
            size = reduce(mul, component.shapes[i_surf])
            arrs.append(vec[cnt : cnt + size].reshape(component.shapes[i_surf]))
        return arrs

    @print_with_time(
        "Linearising aerodynamic system...", "Linearisation complete in {:.2f} seconds."
    )
    def linearise(self) -> LinearSystem:
        def _make_e_mat(zeta_bs: ArrayList) -> Array:
            r"""
            Matrix for [A(zeta_c, zeta_b) \cdot n]^{-1}
            [n_surf][zeta_m, zeta_n, 3] -> [m_tot*n_tot, m_tot*n_tot]
            """
            zeta_cs = get_c(zeta_bs)
            ns = get_nc(zeta_bs)
            aic_sys = compute_aic_sys_assembled(zeta_cs, zeta_bs, self.kernels_b, ns)
            return jnp.linalg.inv(aic_sys)

        def _make_v_bc(
            zeta_bs: ArrayList,
            zeta_ws: ArrayList,
            gamma_ws: ArrayList,
            zeta_bs_dot: ArrayList,
        ) -> Array:
            r"""
            Boundary condition velocity at collocation points.
            :param zeta_bs: Bound vertex positions at time=n+1, [n_surf][zeta_m, zeta_n, 3]
            :param zeta_ws: Wake vertex positions at time=n+1, [n_surf][zeta_m_star, zeta_n, 3]
            :param gamma_ws: Wake strengths at time=n+1, [n_surf][m_star, n, 3
            :param zeta_bs_dot: Wake vertex velocities at time=n+1, [n_surf][zetas_m, zeta_n, 3]
            :return: Boundary condition velocity at collocation points, [m_tot*n_tot]
            """
            # all values given at time=n+1
            zeta_cs = get_c(zeta_bs)
            zeta_cs_dot = get_c(zeta_bs_dot)

            ns = get_nc(zeta_bs)
            aic_w = compute_aic_sys_assembled(
                zeta_cs, zeta_ws, self.kernels_w, ns
            )  # [m_tot*n_tot, m_star_tot*n_tot]

            v_zeta_n = ArrayList.einsum(
                "ijk,ijk->ij",
                self.flowfield0.surf_vmap_call(zeta_cs, jnp.array(self.t0))
                - zeta_cs_dot,
                ns,
            )

            return aic_w @ gamma_ws.flatten() + v_zeta_n.flatten()

        def _v_flow(
            x: Array,
            gamma_b: Optional[ArrayList],
            gamma_w: Optional[ArrayList],
            zeta_b: Optional[ArrayList],
            zeta_w: Optional[ArrayList],
        ) -> Array:
            # sample flowfield
            v_x = self.flowfield0.vmap_call(x, jnp.array(self.t0))

            # add influence from elements if gamma is provided
            if gamma_b is not None and gamma_w is not None:
                vertex_influence = compute_aic_sys_assembled(
                    [x], [*zeta_b, *zeta_w], [*self.kernels_b, *self.kernels_w], None
                )

                # add influence from panels
                v_x += jnp.einsum(
                    "ijk,j->ik",
                    vertex_influence,
                    jnp.concatenate([gamma_b.flatten(), gamma_w.flatten()]),
                ).reshape(*x.shape)
            return v_x

        def _propagate_linear_wake(
            u_np1: InputUnflattened, x_n: StateUnflattened
        ) -> tuple[Optional[ArrayList], ArrayList]:
            u_np1_tot = self.get_total_input(u_np1)
            x_n_tot = self.get_total_state(x_n)

            def _v_wake_prop(x: Array) -> Array:
                # flow is from previous state
                return _v_flow(
                    x,
                    x_n_tot.gamma_b if self.free_wake else None,
                    x_n_tot.gamma_w if self.free_wake else None,
                    x_n_tot.zeta_b if self.free_wake else None,
                    x_n_tot.zeta_w if self.free_wake else None,
                )

            # use wake propagation routines from nonlinear case, as they should be equivalent
            zeta_w_np1_tot, gamma_w_np1_tot = propagate_wake(
                x_n_tot.gamma_b,
                x_n_tot.gamma_w,
                u_np1_tot.zeta_b if self.prescribed_wake else self.zeta0_b,
                x_n_tot.zeta_w if self.prescribed_wake else self.zeta0_w,
                self.delta_w,
                _v_wake_prop,
                self.dt,
                not self.prescribed_wake,
            )

            # obtain the delta for the linear system
            d_gamma_w_np1 = gamma_w_np1_tot - self.gamma0_w
            d_zeta_w_np1 = (
                (zeta_w_np1_tot - self.zeta0_w) if self.prescribed_wake else None
            )

            # add pertubations from input_ velocities
            if self.prescribed_wake and self.wake_upwash:
                nu_w_n = u_np1_tot.nu_w  # TODO: shift in time
                d_zeta_w_np1 += nu_w_n * self.dt

            return d_zeta_w_np1, d_gamma_w_np1

        def _get_dn(d_zeta_b: Sequence[Array]) -> Sequence[Array]:
            zeta_b_full = d_zeta_b + self.zeta0_b
            n_full = get_nc(zeta_b_full)
            return n_full - self.n0

        # boundary condition velocity and its derivatives [n_c]
        v_bc0 = _make_v_bc(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)

        def d_v_bc_d_zeta_b(d_zeta_b: ArrayList) -> Array:
            def func(dzb_: ArrayList) -> Array:
                return _make_v_bc(
                    ArrayList(dzb_), self.zeta0_w, self.gamma0_w, self.zeta0_b_dot
                )

            primals, tangents = jax.jvp(func, [self.zeta0_b], [d_zeta_b])
            return ArrayList(tangents).flatten()

        def d_v_bc_d_zeta_w(d_zeta_w: ArrayList) -> Array:
            def func(dzw_: ArrayList) -> Array:
                return _make_v_bc(self.zeta0_b, dzw_, self.gamma0_w, self.zeta0_b_dot)

            primals, tangents = jax.jvp(func, [self.zeta0_w], [d_zeta_w])
            return ArrayList(tangents).flatten()

        def d_v_bc_d_gamma_w(d_gamma_w: ArrayList) -> Array:
            def func(dgw_: ArrayList) -> Array:
                return _make_v_bc(self.zeta0_b, self.zeta0_w, dgw_, self.zeta0_b_dot)

            primals, tangents = jax.jvp(func, [self.gamma0_w], [d_gamma_w])
            return ArrayList(tangents).flatten()

        # e matrix and its derivative
        # [n_c, n_c]
        e0 = _make_e_mat(self.zeta0_b)

        def d_e_d_zeta_b(d_zeta_b: ArrayList) -> Array:
            def func(dzb_: ArrayList) -> Array:
                return _make_e_mat(ArrayList(dzb_))

            primals, tangents = jax.jvp(func, [self.zeta0_b], [d_zeta_b])
            return sum(tangents)

        @jit
        def _a_func(x_n_vec: Array) -> Array:
            x_n = self._unpack_state_vector(x_n_vec)

            # set previous bound circulation
            d_gamma_bm1_np1 = x_n.gamma_b  # working

            # no contribution to bound grid
            d_zeta_b_np1 = (
                ArrayList(
                    [jnp.zeros(shapes) for shapes in self.state_slices.zeta_b.shapes]
                )
                if self.prescribed_wake
                else None
            )

            # use wake routines to get new wake circulation, factoring in variable discretization
            d_zeta_w_np1, d_gamma_w_np1 = _propagate_linear_wake(
                self.get_zero_input(), x_n
            )

            # influence of states on bound circulation
            d_v_bc = d_v_bc_d_gamma_w(d_gamma_w_np1)

            if self.prescribed_wake:
                d_v_bc += d_v_bc_d_zeta_w(d_zeta_w_np1)

            # resulting bound circulation perturbation
            d_gamma_b_np1 = self._unflatten_subvec(
                -e0 @ d_v_bc, self.state_slices.gamma_b
            )

            d_gamma_b_dot_np1 = (
                (d_gamma_b_np1 - d_gamma_bm1_np1) / self.dt
                if self.gamma_dot_state
                else None
            )

            state_np1 = StateUnflattened(
                d_gamma_b_np1,
                d_gamma_w_np1,
                d_gamma_bm1_np1,
                d_gamma_b_dot_np1,
                d_zeta_w_np1,
                d_zeta_b_np1,
            )
            return self._pack_state_vector(state_np1)

        @jit
        def _b_func(u_np1_vec: Array) -> Array:
            u_np1 = self._unpack_input_vector(u_np1_vec)

            # influence of grid perturbations on wake influence
            d_v_bc = d_v_bc_d_zeta_b(u_np1.zeta_b)

            # perturbations in flow and bound grid at zeta_b
            d_n = _get_dn(u_np1.zeta_b)
            d_zeta_dot_c = get_c(u_np1.zeta_b_dot)
            zeta0_c_dot = get_c(self.zeta0_b_dot)

            if self.bound_upwash:
                d_nu_c = get_c(u_np1.nu_b)
                d_v_bc += (
                    ArrayList.einsum("ijk,ijk->ij", d_nu_c - d_zeta_dot_c, self.n0)
                    + ArrayList.einsum("ijk,ijk->ij", -zeta0_c_dot, d_n)
                ).flatten()
            else:
                d_v_bc += (
                    ArrayList.einsum("ijk,ijk->ij", -d_zeta_dot_c, self.n0)
                    + ArrayList.einsum("ijk,ijk->ij", -zeta0_c_dot, d_n)
                ).flatten()

            d_gamma_b_np1_vec = -e0 @ d_v_bc

            # pertubations in E matrix [n_c, n_c]
            d_e = d_e_d_zeta_b(u_np1.zeta_b)
            d_gamma_b_np1_vec -= d_e @ v_bc0

            # pertubations in solve matrix
            d_gamma_b_np1 = self._unflatten_subvec(
                d_gamma_b_np1_vec, self.state_slices.gamma_b
            )

            # pertubations in wake
            d_zeta_w_np1, d_gamma_w_np1 = _propagate_linear_wake(
                u_np1, self.get_zero_state()
            )

            # pertubations in gamma dot state
            d_gamma_dot_np1 = d_gamma_b_np1 / self.dt if self.gamma_dot_state else None

            state_np1 = StateUnflattened(
                d_gamma_b_np1,
                d_gamma_w_np1,
                ArrayList(
                    [jnp.zeros(shapes) for shapes in self.state_slices.gamma_bm1.shapes]
                )
                if self.unsteady_force
                else None,
                d_gamma_dot_np1,
                d_zeta_w_np1,
                u_np1.zeta_b,
            )

            return self._pack_state_vector(state_np1)

        @jit
        def _c_func(x_n_vec: Array) -> Array:
            x_n = self._unpack_state_vector(x_n_vec)

            if self.unsteady_force:
                d_gamma_dot_n = (
                    x_n.gamma_b_dot
                    if self.gamma_dot_state
                    else (x_n.gamma_b - x_n.gamma_bm1) / self.dt
                )
                d_f_unsteady_n = self.flowfield0.rho * ArrayList(
                    [
                        split_to_vertex(arr, (0, 1))
                        for arr in ArrayList.einsum(
                            "ij,ijk->ijk", d_gamma_dot_n, self.n0
                        )
                    ]
                )
            else:
                d_f_unsteady_n = None

            def steady_forcing_c(
                gamma_b: ArrayList,
                gamma_w: ArrayList,
                zeta_w: Optional[ArrayList] = None,
            ):
                def _v_forcing(x: Array) -> Array:
                    return _v_flow(
                        x,
                        gamma_b,
                        gamma_w,
                        self.zeta0_b,
                        zeta_w if zeta_w is not None else self.zeta0_w,
                    )

                return steady_forcing(
                    self.zeta0_b,
                    self.zeta0_b_dot,
                    gamma_b,
                    gamma_w,
                    _v_forcing,
                    None,
                    self.flowfield0.rho,
                )

            d_f_steady_n = jax.jvp(
                steady_forcing_c,
                [self.gamma0_b, self.gamma0_w, self.zeta0_w]
                if self.prescribed_wake
                else [self.gamma0_b, self.gamma0_w],
                [x_n.gamma_b, x_n.gamma_w, x_n.zeta_w]
                if self.prescribed_wake
                else [x_n.gamma_b, x_n.gamma_w],
            )[1]

            return self._pack_output_vector(
                OutputUnflattened(d_f_steady_n, d_f_unsteady_n)
            )

        @jit
        def _d_func(u_n_vec: Array) -> Array:
            u_n = self._unpack_input_vector(u_n_vec)

            def steady_forcing_d(
                zeta_b: ArrayList, zeta_b_dot: ArrayList, nu_b: Optional[ArrayList]
            ) -> ArrayList:
                def _v_forcing(x: Array) -> Array:
                    return _v_flow(
                        x, self.gamma0_b, self.gamma0_w, zeta_b, self.zeta0_w
                    )

                return steady_forcing(
                    zeta_b,
                    zeta_b_dot,
                    self.gamma0_b,
                    self.gamma0_w,
                    _v_forcing,
                    nu_b,
                    self.flowfield0.rho,
                )

            if self.bound_upwash:
                d_f_steady_n = jax.jvp(
                    steady_forcing_d,
                    [
                        self.zeta0_b,
                        self.zeta0_b_dot,
                        ArrayList.zeros_like(self.zeta0_b),
                    ],
                    [u_n.zeta_b, u_n.zeta_b_dot, u_n.nu_b],
                )[1]
            else:
                d_f_steady_n = jax.jvp(
                    lambda zb, zbd: steady_forcing_d(zb, zbd, None),
                    [self.zeta0_b, self.zeta0_b_dot],
                    [u_n.zeta_b, u_n.zeta_b_dot],
                )[1]

            if self.unsteady_force:
                # no contribution from input_ to unsteady forces, assuming that gamma0_b_dot is zero
                d_f_unsteady_n = ArrayList.zeros_like(d_f_steady_n)
            else:
                d_f_unsteady_n = None

            return self._pack_output_vector(
                OutputUnflattened(d_f_steady_n, d_f_unsteady_n)
            )

        a = LinearOperator(_a_func, shape=(self.n_states, self.n_states))
        b = LinearOperator(_b_func, shape=(self.n_states, self.n_inputs))
        c = LinearOperator(_c_func, shape=(self.n_outputs, self.n_states))
        d = LinearOperator(_d_func, shape=(self.n_outputs, self.n_inputs))

        return LinearSystem(a, b, c, d)

    @replace_self
    def run(
        self,
        u: InputUnflattened,
        x0: Optional[StateUnflattened] = None,
        flowfield: Optional[FlowField] = None,
        use_matrix=False,
    ) -> Self:
        r"""
        Run the linear system for one time step.
        :param u: Input perturbations at time=n+1
        :param x0: State perturbations at time=n
        :return: Output perturbations at time=n+1
        """
        if self.prescribed_wake and self.sys.removed_u_np1:
            warn(
                "Wake pertubations coordinates at the trailing edge are zero when removing u_np1 from the system."
            )

        if x0 is None:
            x0_vec = None
        else:
            x0_vec = self._pack_state_vector_t(x0)

        self._n_tstep: int = u.zeta_b[0].shape[
            0
        ]  # number of time steps from first surface, first entry
        self._t = self.t0 + jnp.arange(0, self._n_tstep) * self.dt  # time vector

        u_tot = u

        if self.bound_upwash and flowfield is None and u_tot.nu_b is None:
            warn(
                "No flowfield or bound upwash perturbations provided. Assuming zero bound upwash perturbations."
            )
            u_tot.nu_b = ArrayList(
                [jnp.zeros((self._n_tstep, *zb.shape)) for zb in self.zeta0_b]
            )

        if self.wake_upwash and flowfield is None and u_tot.nu_w is None:
            warn(
                "No flowfield or wake upwash perturbations provided. Assuming zero wake upwash perturbations."
            )
            u_tot.nu_w = ArrayList(
                [jnp.zeros((self._n_tstep, *zw.shape)) for zw in self.zeta0_w]
            )

        # add flowfield contributions to input upwash if provided
        if flowfield is not None:
            if self.bound_upwash:
                nu_b_flow = ArrayList([])
                for i_surf in range(self.n_surf):
                    nu_b_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.zeta0_b[i_surf], self.t
                        )
                        - flowfield.vmap_call(self.zeta0_b[i_surf], self.t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_b is None:
                    u_tot.nu_b = nu_b_flow
                else:
                    u_tot.nu_b += nu_b_flow
            if self.wake_upwash:
                nu_w_flow = ArrayList([])
                for i_surf in range(self.n_surf):
                    nu_w_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.zeta0_w[i_surf], self.t
                        )
                        - flowfield.vmap_call(self.zeta0_w[i_surf], self.t[0])[
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

        # save results to object
        self._u_t = u
        self._x_t = self._unpack_state_vector_t(x_t)
        self._y_t = self._unpack_output_vector_t(y_t)
        self._u_t_tot = self.get_total_input_t(self._u_t)
        self._x_t_tot = self.get_total_state_t(self._x_t)
        self._y_t_tot = self.get_total_output_t(self._y_t)

        jax.block_until_ready(self)
        return self

    @print_with_time(
        "Computing eigenvalues of linear system...",
        "Eigenvalues computed in {:.2f} seconds.",
    )
    def eigenvalues(self, to_components: bool = True) -> Array:
        r"""
        Compute stability eigenvalues of the linear system A matrix.
        :return: Eigenvalues of the A matrix
        """
        evals = jnp.linalg.eigvals(self.sys.a.matrix)
        if to_components:
            return jnp.stack((evals.real, evals.imag), axis=-1)
        else:
            return evals

    def __getitem__(self, i_ts: int) -> AeroSnapshot:
        r"""
        Get snapshot of aerodynamic surface at a single time step
        :param i_ts: Timestep index
        :return: AeroSnapshot at specified time step
        """

        if i_ts < 0 or i_ts >= self._n_tstep:
            raise IndexError("Timestep index out of range")

        # always exist
        zeta_b_tot = ArrayList(
            [self.u_t_tot.zeta_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        zeta_b_dot_tot = ArrayList(
            [
                self.u_t_tot.zeta_b_dot[i_surf][i_ts, ...]
                for i_surf in range(self.n_surf)
            ]
        )
        gamma_b_tot = ArrayList(
            [self.x_t_tot.gamma_b[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        gamma_w_tot = ArrayList(
            [self.x_t_tot.gamma_w[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )
        f_steady_tot = ArrayList(
            [self.y_t_tot.f_steady[i_surf][i_ts, ...] for i_surf in range(self.n_surf)]
        )

        # optional
        gamma_b_dot_tot = (
            ArrayList(
                [
                    self.x_t_tot.gamma_b_dot[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.gamma_dot_state
            else self.gamma0_b_dot
        )
        zeta_w_tot = (
            ArrayList(
                [
                    self.x_t_tot.zeta_w[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.prescribed_wake
            else self.zeta0_w
        )
        f_unsteady_tot = (
            ArrayList(
                [
                    self.y_t_tot.f_unsteady[i_surf][i_ts, ...]
                    for i_surf in range(self.n_surf)
                ]
            )
            if self.unsteady_force
            else self.f_unsteady0
        )

        return AeroSnapshot(
            zeta_b=zeta_b_tot,
            zeta_b_dot=zeta_b_dot_tot,
            zeta_w=zeta_w_tot,
            gamma_b=gamma_b_tot,
            gamma_b_dot=gamma_b_dot_tot,
            gamma_w=gamma_w_tot,
            f_steady=f_steady_tot,
            f_unsteady=f_unsteady_tot,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t[i_ts],
            n_surf=self.n_surf,
        )

    def reference_snapshot(self) -> AeroSnapshot:
        return AeroSnapshot(
            zeta_b=self.zeta0_b,
            zeta_b_dot=self.zeta0_b_dot,
            zeta_w=self.zeta0_w,
            gamma_b=self.gamma0_b,
            gamma_b_dot=self.gamma0_b_dot,
            gamma_w=self.gamma0_w,
            f_steady=self.f_steady0,
            f_unsteady=self.f_unsteady0,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=-1,
            t=jnp.zeros(()),
            n_surf=self.n_surf,
        )

    @print_with_time(
        "Plotting linear aerodynamic grid...",
        "Linear aerodynamic grid plotted in {:.2f} seconds.",
    )
    def plot(
        self,
        directory: PathLike,
        index: Optional[slice | Sequence[int] | int | Array] = None,
        plot_wake: bool = True,
    ) -> None:
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep_tot)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index,)
        elif index is None:
            index_ = jnp.arange(self.n_tstep_tot)
        else:
            raise TypeError("index must be a slices, sequence of ints, or Array")

        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)

        paths: list[Sequence[Path]] = []
        for i_ts in index_:
            snapshot = self[i_ts]
            paths.append(snapshot.plot(directory, plot_wake=plot_wake))

        for i_surf in range(2 * self.n_surf):
            try:
                surf_paths = [paths[i][i_surf] for i in range(len(index_))]
                name = (self.surf_b_names + self.surf_w_names)[i_surf] + "_ts"
                write_pvd(directory, name, surf_paths, list(self.t[index_]))
            except IndexError:
                pass

    @print_with_time(
        "Plotting linear reference aerodynamic grid...",
        "Reference aerodynamic grid plotted in {:.2f} seconds.",
    )
    def plot_reference(
        self, directory: PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), plot_wake=plot_wake
        )
