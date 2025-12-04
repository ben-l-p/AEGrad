from __future__ import annotations
from jax import Array
import jax
import jax.numpy as jnp
from typing import Sequence, TYPE_CHECKING, Optional
from functools import reduce
from operator import mul
from enum import Enum

from aegrad.aero.data_structures import (AeroSnapshot, InputSlices, StateSlices, OutputSlices, LinearComponent,
                                         _SliceEntry, InputUnflattened, StateUnflattened, OutputUnflattened)
from aegrad.aero.uvlm_utils import get_c, get_nc, propagate_wake
from aegrad.algebra.base import LinearOperator
from aegrad.array_utils import flatten_to_1d
from aegrad.aero.aic import compute_aic_sys_assembled
from aegrad.aero.flowfields import FlowField
from aegrad.aero.kernels import KernelFunction
from aegrad.utils import shallow_asdict

if TYPE_CHECKING:
    from aegrad.aero.case import AeroCase

class LinearWakeType(Enum):
    # (is prescribed, is free)
    FROZEN = (False, False)
    PRESCRIBED = (True, False)
    FREE = (True, True)

class LinearAero:
    def __init__(self,
                 case: AeroCase,
                 reference: AeroSnapshot,
                 wake_type: LinearWakeType = LinearWakeType.FROZEN,
                 bound_upwash: bool = True,
                 wake_upwash: bool = True,
                 unsteady_force: bool = True):

        # options
        self.prescribed_wake, self.free_wake = wake_type.value
        self.unsteady_force: bool = unsteady_force
        self.bound_upwash: bool = bound_upwash
        self.wake_upwash: bool = wake_upwash

        # time info
        self.dt: Array = case.dt
        self.t: Array = reference.t

        # reference state
        self.n_surf: int = reference.n_surf
        self.zeta0_b: Sequence[Array] = reference.zeta_b
        self.zeta0_b_dot: Sequence[Array] = reference.zeta_b_dot
        self.zeta0_w: Sequence[Array] = reference.zeta_w
        self.gamma0_b: Sequence[Array] = reference.gamma_b
        self.gamma0_w: Sequence[Array] = reference.gamma_w
        self.f_steady0: Sequence[Array] = reference.f_steady
        self.f_unsteady0: Sequence[Array] = reference.f_unsteady
        self.c0: Sequence[Array] = get_c(reference.zeta_b)
        self.n0: Sequence[Array] = get_nc(reference.zeta_b)
        self.flowfield0: FlowField = case.flowfield

        # baseline shapes
        self.zeta_b_shapes: Sequence[tuple[int, ...]] = [arr.shape for arr in self.zeta0_b]
        self.zeta_w_shapes: Sequence[tuple[int, ...]] = [arr.shape for arr in self.zeta0_w]
        self.gamma_b_shapes: Sequence[tuple[int, ...]] = [arr.shape for arr in self.gamma0_b]
        self.gamma_w_shapes: Sequence[tuple[int, ...]] = [arr.shape for arr in self.gamma0_w]

        # slices of individial surface components in full vector
        self.input_slices, self.n_inputs = self.make_input_slices()
        self.state_slices, self.n_states = self.make_state_slices()
        self.output_slices, self.n_outputs = self.make_output_slices()

        # kernels
        self.kernels_b: Sequence[KernelFunction] = case.kernels_b
        self.kernels_w: Sequence[KernelFunction] = case.kernels_w

        # wake propagation deltas
        self.delta_w: Sequence[Optional[Array]] = case.delta_w

        # linear operators for system
        self._a_mat, self._b_mat, self._c_mat, self._d_mat = self.linearise()

    def get_reference_inputs(self) -> InputUnflattened:
        return InputUnflattened(self.zeta0_b, self.zeta0_b_dot,
                                [jnp.zeros_like(arr) for arr in self.zeta0_b] if self.bound_upwash else None,
                                [jnp.zeros_like(arr) for arr in self.zeta0_w] if self.wake_upwash else None)

    def get_reference_states(self) -> StateUnflattened:
        return StateUnflattened(
            self.gamma0_b, self.gamma0_w, self.gamma0_b if self.unsteady_force else None,
            self.zeta0_w if self.prescribed_wake else None,
            self.zeta0_b if self.free_wake else None,
        )

    def get_reference_outputs(self) -> OutputUnflattened:
        return OutputUnflattened(
            self.f_steady0,
            self.f_unsteady0 if self.unsteady_force else None
        )

    @staticmethod
    def _make_slices[T](slice_entries: Sequence[_SliceEntry],
                        cls: type[T]) -> tuple[T, int]:
        r"""
        Helper function to create slices classes for the vectors, and count the number of elements.
        Blocks should be passed in the order they are in the dataclass.
        """
        # make slices
        cnt = 0
        out_dict = {}
        for entry in slice_entries:
            if not entry.enabled:    # if disabled
                out_dict[entry.name] = LinearComponent(False, None, None)
            else:
                slices = []
                for size in [reduce(mul, shape) for shape in entry.shapes]:
                    slices.append(slice(cnt, cnt + size))
                    cnt += size
                out_dict[entry.name] = LinearComponent(True, slices, entry.shapes)
        return cls(**out_dict), cnt

    def make_input_slices(self) -> tuple[InputSlices, int]:
        slice_entries = (_SliceEntry("zeta_b", True, self.zeta_b_shapes),
                    _SliceEntry("zeta_b_dot", True, self.zeta_b_shapes),
                    _SliceEntry("nu_c", *((True, self.zeta_b_shapes) if self.bound_upwash else (False, None))),
                    _SliceEntry("nu_w", *((True, self.zeta_w_shapes) if self.wake_upwash else (False, None))))
        return self._make_slices(slice_entries, InputSlices)

    def make_state_slices(self) -> tuple[StateSlices, int]:
        slice_entries = (
            _SliceEntry("gamma_b", True, self.gamma_b_shapes),
            _SliceEntry("gamma_w", True, self.gamma_w_shapes),
            _SliceEntry("gamma_bm1", *((True, self.gamma_b_shapes) if self.unsteady_force else (False, None))),
            _SliceEntry("zeta_w", *((True, self.zeta_w_shapes) if self.prescribed_wake else (False, None))),
            _SliceEntry("zeta_b", *((True, self.zeta_b_shapes) if self.free_wake else (False, None))))
        return self._make_slices(slice_entries, StateSlices)

    def make_output_slices(self) -> tuple[OutputSlices, int]:
        slice_entries = (
            _SliceEntry("f_steady", True, self.zeta_b_shapes),
            _SliceEntry("f_unsteady", *((True, self.zeta_b_shapes) if self.unsteady_force else (False, None))))
        return self._make_slices(slice_entries, OutputSlices)

    def _unpack_vector(self, x: Array, slices: dict[str, LinearComponent]) -> dict[str, Optional[Sequence[Array]]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                out[name]  = [x[entry.slices[i_surf]].reshape(entry.shapes[i_surf]) for i_surf in range(self.n_surf)]
        return out

    def unpack_input_vector(self, u: Array) -> InputUnflattened:
        return InputUnflattened(**self._unpack_vector(u, shallow_asdict(self.input_slices)))

    def unpack_state_vector(self, x: Array) -> StateUnflattened:
        return StateUnflattened(**self._unpack_vector(x, shallow_asdict(self.state_slices)))

    def unpack_output_vector(self, y: Array) -> OutputUnflattened:
        return OutputUnflattened(**self._unpack_vector(y, shallow_asdict(self.output_slices)))

    def _pack_vector(self, slices: dict[str, LinearComponent], vec_length: int, arrs: dict[str, Optional[Sequence[Array]]]) -> Array:
        vec = jnp.zeros(vec_length)
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.n_surf):
                    vec = vec.at[entry.slices[i_surf]].set(arrs[name][i_surf].ravel())
        return vec

    def pack_input_vector(self, u_input: InputUnflattened) -> Array:
        return self._pack_vector(shallow_asdict(self.input_slices), self.n_inputs, shallow_asdict(u_input))

    def pack_state_vector(self, x_state: StateUnflattened) -> Array:
        return self._pack_vector(shallow_asdict(self.state_slices), self.n_states, shallow_asdict(x_state))

    def pack_output_vector(self, y_output: OutputUnflattened) -> Array:
        return self._pack_vector(shallow_asdict(self.output_slices), self.n_outputs, shallow_asdict(y_output))

    def _get_total(self,
                   input: dict[str, Optional[Sequence[Array]]],
                   reference: dict[str, Optional[Sequence[Array]]]) -> dict[str, Optional[Sequence[Array]]]:
        out = {}
        for name, entry in reference.items():
            if entry is None:
                out[name] = None
            else:
                arrs = []
                for i_surf in range(self.n_surf):
                    arrs.append(reference[name][i_surf] + input[name][i_surf])
                out[name] = arrs
        return out

    def get_total_input(self, u: InputUnflattened) -> InputUnflattened:
        return InputUnflattened(**self._get_total(shallow_asdict(u), shallow_asdict(self.get_reference_inputs())))

    def get_total_state(self, x: StateUnflattened) -> StateUnflattened:
        return StateUnflattened(**self._get_total(shallow_asdict(x), shallow_asdict(self.get_reference_states())))

    def get_total_output(self, y: OutputUnflattened) -> OutputUnflattened:
        return OutputUnflattened(**self._get_total(shallow_asdict(y), shallow_asdict(self.get_reference_outputs())))

    def _get_zero(self, slices: dict[str, LinearComponent]) -> dict[str, Optional[Sequence[Array]]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                arrs = []
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

    def unflatten_subvec(self, vec: Array, component: LinearComponent) -> Sequence[Array]:
        arrs = []
        cnt = 0
        for i_surf in range(self.n_surf):
            size = reduce(mul, component.shapes[i_surf])
            arrs.append(vec[cnt:cnt+size].reshape(component.shapes[i_surf]))
        return arrs


    def linearise(self) -> tuple[LinearOperator, LinearOperator, LinearOperator, LinearOperator]:

        def make_e_mat(zeta_bs: Sequence[Array]) -> Array:
            r"""
            Matrix for [A(zeta_c, zeta_b) \cdot n]^{-1}
            [n_surf][zeta_m, zeta_n, 3] -> [m_tot*n_tot, m_tot*n_tot]
            """
            zeta_cs = get_c(zeta_bs)
            ns = get_nc(zeta_bs)
            aic_sys = compute_aic_sys_assembled(
                zeta_cs, zeta_bs, self.kernels_b, ns)
            return jnp.linalg.inv(aic_sys)

        def make_v_bc(zeta_bs: Sequence[Array],
                   zeta_ws: Sequence[Array],
                   gamma_ws: Sequence[Array],
                   zeta_bs_dot: Sequence[Array]) -> Array:
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
            aic_w = compute_aic_sys_assembled(zeta_cs, zeta_ws, self.kernels_w, ns)  # [m_tot*n_tot, m_star_tot*n_tot]

            v_zeta_n = [jnp.einsum('ijk,ijk->ij', (self.flowfield0.vmap_call(x_, jnp.array(self.t)) - x_dot), n_)
                        for x_, x_dot, n_ in zip(zeta_cs, zeta_cs_dot, ns)]
            return aic_w @ flatten_to_1d(gamma_ws) + flatten_to_1d(v_zeta_n)

        def propagate_linear_wake(u_n: InputUnflattened, x_n: StateUnflattened) -> tuple[Optional[Sequence[Array]], Sequence[Array]]:
            u_n_tot = self.get_total_input(u_n)
            x_n_tot = self.get_total_state(x_n)

            def v_wake_prop(x: Array) -> Array:
                v_x = self.flowfield0.vmap_call(x, jnp.array(self.t))

                # optionally include circulation pertubations in the wake convection velocity
                if self.free_wake:
                    vertex_influence = compute_aic_sys_assembled([x],
                                                             [*x_n_tot.zeta_b, *x_n_tot.zeta_b],
                                                            [*self.kernels_b, *self.kernels_w],
                                                             None)

                return v_x



            # use wake propagation routines from nonlinear case, as they should be equivalent
            zeta_w_np1_tot, gamma_w_np1_tot = propagate_wake(
                            x_n_tot.gamma_b,
                            x_n_tot.gamma_w,
                            x_n_tot.zeta_b if self.prescribed_wake else self.zeta0_b,
                            x_n_tot.zeta_w if self.prescribed_wake else self.zeta0_w,
                           self.delta_w,
                           v_wake_prop,
                           self.dt,
                           not self.prescribed_wake,
                           )

            # obtain the delta for the linear system
            d_gamma_w_np1 = [new - base for new, base in zip(gamma_w_np1_tot, self.gamma0_w)]
            d_zeta_w_np1 = [new - base for new, base in zip(zeta_w_np1_tot, self.zeta0_w)] if self.prescribed_wake else None
            return d_zeta_w_np1, d_gamma_w_np1


        # boundary condition velocity and its derivatives [n_c]
        v_bc0 = make_v_bc(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)

        # [n_surf][n_c, zeta_b_m, zeta_b_n, 3]
        d_v_bc_d_zeta_b = jax.jacobian(make_v_bc, argnums=0)(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)

        # [n_surf][n_c, zeta_w_m, zeta_b_n, 3]
        if self.prescribed_wake:
            d_v_bc_d_zeta_w = jax.jacobian(make_v_bc, argnums=1)(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)
        else:
            d_v_bc_d_zeta_w = None

        # [n_surf][n_c, m_star, n]
        d_v_bc_d_gamma_w = jax.jacobian(make_v_bc, argnums=2)(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)

        # [n_surf][n_c, zeta_b_m, zeta_b_n, 3]
        d_v_bc_d_zeta_b_dot = jax.jacobian(make_v_bc, argnums=3)(self.zeta0_b, self.zeta0_w, self.gamma0_w, self.zeta0_b_dot)

        # e matrix and its derivative
        # [n_c, n_c]
        e0 = make_e_mat(self.zeta0_b)

        # [n_surf][n_c, n_c, zeta_b_m, zeta_b_n, 3]
        d_e_d_zeta_b = jax.jacobian(make_e_mat, argnums=0)(self.zeta0_b)

        def a_func(x_n_vec: Array) -> Array:
            x_n = self.unpack_state_vector(x_n_vec)

            # set previous bound circulation
            d_gamma_bm1_np1 = x_n.gamma_b   # working

            # no contribution to bound grid
            d_zeta_b_np1 = [jnp.zeros(shapes) for shapes in self.state_slices.zeta_b.shapes] if self.prescribed_wake else None

            # use wake routines to get new wake circulation, factoring in variable discretization
            d_zeta_w_np1, d_gamma_w_np1 = propagate_linear_wake(self.get_zero_input(), x_n)

            # influence of states on bound circulation
            d_v_bc = jnp.concatenate([jnp.einsum('ijk,jk->i', d_v_bc_d_gamma_w[i_surf], d_gamma_w_np1[i_surf]) for i_surf in range(self.n_surf)])
            if self.prescribed_wake:
                d_v_bc += jnp.concatenate(
                    [
                        jnp.einsum(
                            "ijkl,jkl->i", d_v_bc_d_zeta_w[i_surf], d_zeta_w_np1[i_surf]
                        )
                        for i_surf in range(self.n_surf)
                    ]
                )

            # resulting bound circulation perturbation
            d_gamma_b_np1 = self.unflatten_subvec(-e0 @ d_v_bc, self.state_slices.gamma_b)

            state_np1 = StateUnflattened(d_gamma_b_np1, d_gamma_w_np1, d_gamma_bm1_np1, d_zeta_w_np1, d_zeta_b_np1)
            return self.pack_state_vector(state_np1)

        def b_func(u_np1_vec: Array) -> Array:
            u_np1 = self.unpack_input_vector(u_np1_vec)

        def c_func(x_n_vec: Array) -> Array:
            x_n = self.unpack_state_vector(x_n_vec)

            pass

        def d_func(u_n_vec: Array) -> Array:
            u_n = self.unpack_state_vector(u_n_vec)

            pass

        a = LinearOperator(jax.jit(a_func), shape=(self.n_states, self.n_states))
        b = LinearOperator(b_func, shape=(self.n_states, self.n_inputs))
        c = LinearOperator(c_func, shape=(self.n_outputs, self.n_states))
        d = LinearOperator(d_func, shape=(self.n_outputs, self.n_inputs))

        return a, b, c, d