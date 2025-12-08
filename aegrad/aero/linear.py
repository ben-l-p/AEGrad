from __future__ import annotations
from jax import Array
import jax
import jax.numpy as jnp
from typing import Sequence, TYPE_CHECKING, Optional
from functools import reduce, singledispatchmethod
from operator import mul
from enum import Enum

from aegrad.aero.data_structures import (AeroSnapshot, InputSlices, StateSlices, OutputSlices, LinearComponent,
                                         _SliceEntry, InputUnflattened, StateUnflattened, OutputUnflattened)
from aegrad.aero.uvlm_utils import get_c, get_nc, propagate_wake, steady_forcing
from aegrad.algebra.linear_operators import LinearOperator, LinearSystem
from aegrad.algebra.array_utils import flatten_to_1d, ArrayList, split_to_vertex
from aegrad.aero.aic import compute_aic_sys_assembled
from aegrad.aero.flowfields import FlowField
from aegrad.aero.kernels import KernelFunction
from aegrad.utils import shallow_asdict
from aegrad.print_output import print_with_time

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
                 wake_type: LinearWakeType = LinearWakeType.FREE,
                 bound_upwash: bool = True,
                 wake_upwash: bool = True,
                 unsteady_force: bool = True,
                 compute_matrices: bool = False):

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
        self.zeta0_b: ArrayList = ArrayList(reference.zeta_b)
        self.zeta0_b_dot: ArrayList = ArrayList(reference.zeta_b_dot)
        self.zeta0_w: ArrayList = ArrayList(reference.zeta_w)
        self.gamma0_b: ArrayList = ArrayList(reference.gamma_b)
        self.gamma0_w: ArrayList = ArrayList(reference.gamma_w)
        self.f_steady0: ArrayList = ArrayList(reference.f_steady)
        self.f_unsteady0: ArrayList = ArrayList(reference.f_unsteady)
        self.c0: ArrayList = get_c(reference.zeta_b)
        self.n0: ArrayList = get_nc(reference.zeta_b)
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
        self.base_sys: LinearSystem = self.linearise()

        # final system - this is overwritten for updating models
        self.sys: LinearSystem = self.base_sys


    def get_reference_inputs(self) -> InputUnflattened:
        return InputUnflattened(self.zeta0_b,
                                self.zeta0_b_dot,
                                [jnp.zeros_like(arr) for arr in self.zeta0_b] if self.bound_upwash else None,
                                [jnp.zeros_like(arr) for arr in self.zeta0_w] if self.wake_upwash else None)

    def get_reference_states(self) -> StateUnflattened:
        return StateUnflattened(
            self.gamma0_b, self.gamma0_w,
            self.gamma0_b if self.unsteady_force else None,
            self.zeta0_w if self.prescribed_wake else None,
            self.zeta0_b if self.prescribed_wake else None,
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
                    _SliceEntry("nu_b", *((True, self.zeta_b_shapes) if self.bound_upwash else (False, None))),
                    _SliceEntry("nu_w", *((True, self.zeta_w_shapes) if self.wake_upwash else (False, None))))
        return self._make_slices(slice_entries, InputSlices)

    def make_state_slices(self) -> tuple[StateSlices, int]:
        slice_entries = (
            _SliceEntry("gamma_b", True, self.gamma_b_shapes),
            _SliceEntry("gamma_w", True, self.gamma_w_shapes),
            _SliceEntry("gamma_bm1", *((True, self.gamma_b_shapes) if self.unsteady_force else (False, None))),
            _SliceEntry("zeta_w", *((True, self.zeta_w_shapes) if self.prescribed_wake else (False, None))),
            _SliceEntry("zeta_b", *((True, self.zeta_b_shapes) if self.prescribed_wake else (False, None))))
        return self._make_slices(slice_entries, StateSlices)

    def make_output_slices(self) -> tuple[OutputSlices, int]:
        slice_entries = (
            _SliceEntry("f_steady", True, self.zeta_b_shapes),
            _SliceEntry("f_unsteady", *((True, self.zeta_b_shapes) if self.unsteady_force else (False, None))))
        return self._make_slices(slice_entries, OutputSlices)

    def _unpack_vector(self, x: Array, slices: dict[str, LinearComponent]) -> dict[str, Optional[ArrayList]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                out[name]  = ArrayList([x[entry.slices[i_surf]].reshape(entry.shapes[i_surf]) for i_surf in range(self.n_surf)])
        return out

    def unpack_input_vector(self, u: Array) -> InputUnflattened:
        return InputUnflattened(**self._unpack_vector(u, shallow_asdict(self.input_slices)))

    def unpack_state_vector(self, x: Array) -> StateUnflattened:
        return StateUnflattened(**self._unpack_vector(x, shallow_asdict(self.state_slices)))

    def unpack_output_vector(self, y: Array) -> OutputUnflattened:
        return OutputUnflattened(**self._unpack_vector(y, shallow_asdict(self.output_slices)))

    def _unpack_vector_t(self, x_t: Array, slices: dict[str, LinearComponent]) -> dict[str, Optional[ArrayList]]:
        out = {}
        for name, entry in slices.items():
            if not entry.enabled:
                out[name] = None
            else:
                out[name]  = ArrayList([x_t[:, entry.slices[i_surf]].reshape(-1, *entry.shapes[i_surf]) for i_surf in range(self.n_surf)])
        return out

    def unpack_input_vector_t(self, u_t: Array) -> InputUnflattened:
        return InputUnflattened(**self._unpack_vector_t(u_t, shallow_asdict(self.input_slices)))

    def unpack_state_vector_t(self, x_t: Array) -> StateUnflattened:
        return StateUnflattened(**self._unpack_vector_t(x_t, shallow_asdict(self.state_slices)))

    def unpack_output_vector_t(self, y_t: Array) -> OutputUnflattened:
        return OutputUnflattened(**self._unpack_vector_t(y_t, shallow_asdict(self.output_slices)))

    def _pack_vector(self, slices: dict[str, LinearComponent], vec_length: int, arrs: dict[str, Optional[ArrayList]]) -> Array:
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

    def _pack_vector_t(self, slices: dict[str, LinearComponent], vec_length: int, arrs: dict[str, Optional[ArrayList]]) -> Array:
        n_tstep = list(arrs.values())[0][0].shape[0]    # find number of timesteps from first surface, first entry
        vec_t = jnp.zeros((n_tstep, vec_length))
        for name, entry in slices.items():
            if entry.enabled:
                for i_surf in range(self.n_surf):
                    vec_t = vec_t.at[:, entry.slices[i_surf]].set(arrs[name][i_surf].reshape(n_tstep, -1))
        return vec_t

    def pack_input_vector_t(self, u_input: InputUnflattened) -> Array:
        return self._pack_vector_t(shallow_asdict(self.input_slices), self.n_inputs, shallow_asdict(u_input))

    def pack_state_vector_t(self, x_state: StateUnflattened) -> Array:
        return self._pack_vector_t(shallow_asdict(self.state_slices), self.n_states, shallow_asdict(x_state))

    def pack_output_vector_t(self, y_output: OutputUnflattened) -> Array:
        return self._pack_vector_t(shallow_asdict(self.output_slices), self.n_outputs, shallow_asdict(y_output))

    def _get_total(self,
                   input: dict[str, Optional[Sequence[Array]]],
                   reference: dict[str, Optional[Sequence[Array]]]) -> dict[str, Optional[Sequence[Array]]]:
        out = {}
        for name, entry in reference.items():
            if entry is None:
                out[name] = None
            else:
                arrs = ArrayList([])
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

    def _get_zero(self, slices: dict[str, LinearComponent]) -> dict[str, Optional[ArrayList]]:
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

    def unflatten_subvec(self, vec: Array, component: LinearComponent) -> ArrayList:
        arrs = ArrayList([])
        cnt = 0
        for i_surf in range(self.n_surf):
            size = reduce(mul, component.shapes[i_surf])
            arrs.append(vec[cnt:cnt+size].reshape(component.shapes[i_surf]))
        return arrs

    @print_with_time("Linearising aerodynamic system...", "Linearisation complete in {:.2f} seconds.")
    def linearise(self) -> LinearSystem:
        def make_e_mat(zeta_bs: ArrayList) -> Array:
            r"""
            Matrix for [A(zeta_c, zeta_b) \cdot n]^{-1}
            [n_surf][zeta_m, zeta_n, 3] -> [m_tot*n_tot, m_tot*n_tot]
            """
            zeta_cs = get_c(zeta_bs)
            ns = get_nc(zeta_bs)
            aic_sys = compute_aic_sys_assembled(
                zeta_cs, zeta_bs, self.kernels_b, ns)
            return jnp.linalg.inv(aic_sys)

        def make_v_bc(zeta_bs: ArrayList,
                   zeta_ws: ArrayList,
                   gamma_ws: ArrayList,
                   zeta_bs_dot: ArrayList) -> Array:
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

            v_zeta_n = ArrayList.einsum("ijk,ijk->ij",
                                        self.flowfield0.surf_vmap_call(zeta_cs, jnp.array(self.t)) - zeta_cs_dot, ns)

            return aic_w @ flatten_to_1d(gamma_ws) + flatten_to_1d(v_zeta_n)

        def v_flow(x: Array,
                    gamma_b: Optional[ArrayList],
                    gamma_w: Optional[ArrayList],
                    zeta_b: Optional[ArrayList],
                    zeta_w: Optional[ArrayList]) -> Array:

            # sample flowfield
            v_x = self.flowfield0.vmap_call(x, jnp.array(self.t))

            # add influence from elements if gamma is provided
            if gamma_b is not None and gamma_w is not None:
                vertex_influence = compute_aic_sys_assembled([x],
                                                             [*zeta_b, *zeta_w],
                                                             [*self.kernels_b, *self.kernels_w],
                                                             None)

                # add influence from panels
                v_x += jnp.einsum('ijk,j->ik', vertex_influence,
                                  jnp.concatenate([gamma_b.flatten(), gamma_w.flatten()])).reshape(*x.shape)
            return v_x

        def propagate_linear_wake(u_n: InputUnflattened, x_n: StateUnflattened) -> tuple[Optional[ArrayList], ArrayList]:
            u_n_tot = self.get_total_input(u_n)
            x_n_tot = self.get_total_state(x_n)

            def v_wake_prop(x: Array) -> Array:
                return v_flow(x,
                              x_n_tot.gamma_b if self.free_wake else None,
                              x_n_tot.gamma_w if self.free_wake else None,
                              u_n_tot.zeta_b if self.free_wake else None,
                              x_n_tot.zeta_w if self.free_wake else None)

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
            d_gamma_w_np1 = gamma_w_np1_tot - self.gamma0_w
            d_zeta_w_np1 = zeta_w_np1_tot - self.zeta0_w if self.prescribed_wake else None

            # add pertubations from input velocities
            if self.prescribed_wake and self.wake_upwash:
                d_zeta_w_np1 += u_n_tot.nu_w * self.dt

            return d_zeta_w_np1, d_gamma_w_np1

        def get_dn(d_zeta_b: Sequence[Array]) -> Sequence[Array]:
            zeta_b_full = d_zeta_b + self.zeta0_b
            n_full = get_nc(zeta_b_full)
            return n_full - self.n0

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
            d_zeta_b_np1 = ArrayList([jnp.zeros(shapes) for shapes in self.state_slices.zeta_b.shapes]) if self.prescribed_wake else None

            # use wake routines to get new wake circulation, factoring in variable discretization
            d_zeta_w_np1, d_gamma_w_np1 = propagate_linear_wake(self.get_zero_input(), x_n)

            # influence of states on bound circulation
            d_v_bc = ArrayList.einsum("ijk,jk->i", d_v_bc_d_gamma_w, d_gamma_w_np1).flatten()

            if self.prescribed_wake:
                d_v_bc += ArrayList.einsum("ijkl,jkl->i", d_v_bc_d_zeta_w, d_zeta_w_np1).flatten()

            # resulting bound circulation perturbation
            d_gamma_b_np1 = self.unflatten_subvec(-e0 @ d_v_bc, self.state_slices.gamma_b)

            state_np1 = StateUnflattened(d_gamma_b_np1, d_gamma_w_np1, d_gamma_bm1_np1, d_zeta_w_np1, d_zeta_b_np1)
            return self.pack_state_vector(state_np1)

        def b_func(u_np1_vec: Array) -> Array:
            u_np1 = self.unpack_input_vector(u_np1_vec)

            # influence of grid pertubations on wake influence
            d_v_bc = ArrayList.einsum("ijkl,jkl->i", d_v_bc_d_zeta_b, u_np1.zeta_b).flatten()

            # pertubations in flow and bound grid at zeta_b
            if self.bound_upwash:
                d_nu_c = get_c(u_np1.nu_b)
                d_n = get_dn(u_np1.zeta_b)
                d_zeta_dot_c = get_c(u_np1.zeta_b_dot)
                zeta0_c_dot = get_c(self.zeta0_b_dot)

                d_v_bc += (ArrayList.einsum('ijk,ijk->ij', d_nu_c - d_zeta_dot_c, self.n0) + ArrayList.einsum('ijk,ijk->ij', -zeta0_c_dot, d_n)).flatten()

            d_gamma_b_np1_vec = -e0 @ d_v_bc

            # pertubations in E matrix [n_c, n_c]
            d_e = sum(ArrayList.einsum("ijklm,klm->ij", d_e_d_zeta_b, u_np1.zeta_b))
            d_gamma_b_np1_vec -= d_e @ v_bc0

            # pertubations in solve matrix
            d_gamma_b_np1 = self.unflatten_subvec(d_gamma_b_np1_vec, self.state_slices.gamma_b)

            # pertubations in wake
            d_zeta_w_np1, d_gamma_w_np1 = propagate_linear_wake(
                u_np1, self.get_zero_state()
            )

            state_np1 = StateUnflattened(
                d_gamma_b_np1,
                d_gamma_w_np1,
                ArrayList([jnp.zeros(shapes) for shapes in self.state_slices.gamma_bm1.shapes]) if self.unsteady_force else None,
                d_zeta_w_np1,
                u_np1.zeta_b,
            )
            return self.pack_state_vector(state_np1)

        def c_func(x_n_vec: Array) -> Array:
            x_n = self.unpack_state_vector(x_n_vec)
            x_n_tot = self.get_total_state(x_n)

            if self.unsteady_force:
                d_f_unsteady_n = split_to_vertex(ArrayList.einsum("ij,ijk->ijk", (x_n.gamma_b - x_n.gamma_bm1) / self.dt, self.n0), (0, 1))
            else:
                d_f_unsteady_n = None

            def v_forcing(x: Array) -> Array:
                return v_flow(x,
                              x_n_tot.gamma_b,
                              x_n_tot.gamma_w,
                              self.zeta0_b,
                              x_n_tot.zeta_w)

            d_f_steady_n = steady_forcing(
                    self.zeta0_b,
                    self.zeta0_b_dot,
                    x_n_tot.gamma_b,
                    x_n_tot.gamma_w,
                    v_forcing,
                None,
                    self.flowfield0.rho,
                ) - self.f_steady0

            return self.pack_output_vector(OutputUnflattened(d_f_steady_n, d_f_unsteady_n))

        def d_func(u_n_vec: Array) -> Array:
            u_n = self.unpack_input_vector(u_n_vec)
            u_n_tot = self.get_total_input(u_n)

            def v_forcing(x: Array) -> Array:
                return v_flow(x,
                              self.gamma0_b,
                              self.gamma0_w,
                              u_n_tot.zeta_b,
                              self.zeta0_w)

            d_f_steady_n = (
                steady_forcing(
                    u_n_tot.zeta_b,
                    u_n_tot.zeta_b_dot,
                    self.gamma0_b,
                    self.gamma0_w,
                    v_forcing,
                    u_n_tot.nu_b if self.bound_upwash else None,
                    self.flowfield0.rho,
                )
                - self.f_steady0
            )

            return self.pack_output_vector(OutputUnflattened(d_f_steady_n, ArrayList.zeros_like(d_f_steady_n)))


        a = LinearOperator(jax.jit(a_func), shape=(self.n_states, self.n_states))
        b = LinearOperator(jax.jit(b_func), shape=(self.n_states, self.n_inputs))
        c = LinearOperator(jax.jit(c_func), shape=(self.n_outputs, self.n_states))
        d = LinearOperator(jax.jit(d_func), shape=(self.n_outputs, self.n_inputs))

        return LinearSystem(a, b, c, d)

    @singledispatchmethod
    def run(self, u: Array, x0: Optional[Array] = None) -> tuple[Array, Array]:
        r"""
        Run the linear system for one time step.
        :param u: Input perturbations at time=n+1
        :param x0: State perturbations at time=n
        :return: Output perturbations at time=n+1
        """
        return self.sys.run(u, x0)

    @run.register
    def run(self, u: InputUnflattened, x0: Optional[StateUnflattened] = None) -> tuple[StateUnflattened, OutputUnflattened]:
        r"""
        Run the linear system for one time step.
        :param u: Input perturbations at time=n+1
        :param x0: State perturbations at time=n
        :return: Output perturbations at time=n+1
        """
        if x0 is None:
            x0_vec = None
        else:
            x0_vec = self.pack_state_vector_t(x0)

        u_vec = self.pack_input_vector_t(u)
        x_vec, y_vec = self.sys.run(u_vec, x0_vec)
        return self.unpack_state_vector_t(x_vec), self.unpack_output_vector_t(y_vec)
