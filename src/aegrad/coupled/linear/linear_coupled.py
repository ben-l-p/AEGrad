from __future__ import annotations
from typing import Any, Callable, Literal, Optional, Sequence, TYPE_CHECKING

from jax import numpy as jnp, Array, vmap

from aegrad.aero.linear import LinearWakeType, LinearUVLM
from aegrad.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroStateUnflattened,
    AeroOutputUnflattened,
)
from aegrad.aero.utils import project_forcing_to_beam
from aegrad.algebra.array_utils import ArrayList, construct_named_block_jacobian
from aegrad.algebra.base import jacrev_custom
from aegrad.algebra.se3 import exp_se3, ha_to_ha_tilde
from aegrad.coupled import StaticAeroelastic
from aegrad.coupled.linear.data_structures import (
    AeroelasticInputUnflattened,
    AeroelasticStateUnflattened,
    AeroelasticOutputUnflattened,
    AeroelasticLinearResult,
)
from aegrad.structure.utils import get_solve_dofs
from aegrad.structure.linear.linear_beam import LinearBeam
from aegrad.utils.constants import BASE_LOBATTO_ORDER
from aegrad.utils.linear import LinearComponent, SliceEntry, LinearModel, LinearSystem
from aegrad.utils.print_utils import print_table_title, print_table_line

if TYPE_CHECKING:
    from aegrad.coupled.coupled import BaseCoupledAeroelastic


class LinearCoupled(
    LinearModel[
        StaticAeroelastic,
        AeroelasticInputUnflattened,
        AeroelasticStateUnflattened,
        AeroelasticLinearResult,
    ]
):
    def __init__(
        self,
        case: BaseCoupledAeroelastic,
        reference: StaticAeroelastic,
        wake_type: LinearWakeType = LinearWakeType.FREE,
        bound_upwash: bool = True,
        wake_upwash: bool = False,
        unsteady_force: bool = True,
        local_forcing: bool = False,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        *,
        skip_checks: bool = False,
    ):

        self._aero = LinearUVLM(
            case=case.aero,
            reference=reference.aero,
            wake_type=wake_type,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
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

        self.unsteady_force: bool = unsteady_force
        self.sys = self.linearise()

    @property
    def reference(self) -> StaticAeroelastic:
        return self._reference

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
            SliceEntry(
                "f_ext_dead",
                True,
                (self.n_beam_dof,),
            ),
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
                "gamma_b_nm1",
                *(
                    (True, reference.aero.gamma_b.shape)
                    if self._aero.unsteady_force
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

    def step(
        self,
        gamma_b_vec: Optional[Array] = None,
        gamma_w_vec: Optional[Array] = None,
        gamma_b_nm1_vec: Optional[Array] = None,
        zeta_w_vec: Optional[Array] = None,
        nu_b_vec: Optional[Array] = None,
        nu_w_vec: Optional[Array] = None,
        f_ext_follower: Optional[Array] = None,
        f_ext_dead: Optional[Array] = None,
        q: Optional[Array] = None,
        q_dot: Optional[Array] = None,
    ) -> tuple[AeroelasticStateUnflattened, AeroelasticOutputUnflattened]:
        r"""
        Step solution from states at timestep n and inputs at timestep n+1 to give states at timestep n+1 and outputs
        at timestep n.
        """
        ref = self.reference

        # unravel vector inputs, falling back to reference values when None
        gamma_b = (
            ArrayList.from_vector(
                vect=gamma_b_vec, arr_list_shapes=ref.aero.gamma_b.shape
            )
            if gamma_b_vec is not None
            else ref.aero.gamma_b
        )
        gamma_w = (
            ArrayList.from_vector(
                vect=gamma_w_vec, arr_list_shapes=ref.aero.gamma_w.shape
            )
            if gamma_w_vec is not None
            else ref.aero.gamma_w
        )
        gamma_b_nm1 = (
            (
                ArrayList.from_vector(
                    vect=gamma_b_nm1_vec, arr_list_shapes=ref.aero.gamma_b.shape
                )
                if gamma_b_nm1_vec is not None
                else ref.aero.gamma_b
            )
            if self._aero.unsteady_force
            else None
        )
        zeta_w = (
            ArrayList.from_vector(
                vect=zeta_w_vec, arr_list_shapes=ref.aero.zeta_w.shape
            )
            if zeta_w_vec is not None
            else ref.aero.zeta_w
        )
        nu_b = (
            (
                ArrayList.from_vector(
                    vect=nu_b_vec, arr_list_shapes=ref.aero.zeta_b.shape
                )
                if nu_b_vec is not None
                else ArrayList.zeros_like(ref.aero.zeta_b)
            )
            if self._aero.bound_upwash
            else None
        )
        nu_w = (
            (
                ArrayList.from_vector(
                    vect=nu_w_vec, arr_list_shapes=ref.aero.zeta_w.shape
                )
                if nu_w_vec is not None
                else ArrayList.zeros_like(ref.aero.zeta_w)
            )
            if self._aero.wake_upwash
            else None
        )
        q = q if q is not None else jnp.zeros(self.n_beam_dof)
        q_dot = q_dot if q_dot is not None else jnp.zeros(self.n_beam_dof)

        # fill in prescribed dofs with zeros
        q_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(q)
            .reshape(self.n_nodes, 6)
        )
        q_dot_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(q_dot)
            .reshape(self.n_nodes, 6)
        )

        # total perturbed coordinates and time derivative
        hg = jnp.einsum("ijk,ikl->ijl", ref.structure.hg, vmap(exp_se3)(q_full))
        hg_dot = jnp.einsum(
            "ijk,ikl->ijl",
            ref.structure.hg,
            vmap(ha_to_ha_tilde)(q_dot_full),
        )

        # aerodynamic grid
        zeta_b = self._aero.case.hg_to_zeta_b(hg_n=hg, cs_ang_n=self._aero.case.cs_ang0)
        zeta_b_dot = self._aero.case.hg_dot_to_zeta_b_dot(
            hg_n=hg,
            hg_dot_n=hg_dot,
            cs_ang_n=self._aero.case.cs_ang0,
            cs_vel_n=self._aero.case.cs_vel0,
        )

        # pass through aerodynamic system
        u_n_aero = AeroInputUnflattened(
            zeta_b=zeta_b, zeta_b_dot=zeta_b_dot, nu_b=nu_b, nu_w=nu_w
        )

        x_n_aero = AeroStateUnflattened(
            gamma_b=gamma_b,
            gamma_w=gamma_w,
            gamma_b_nm1=gamma_b_nm1,
            zeta_w=zeta_w if self._aero.prescribed_wake else None,
            zeta_b=zeta_b if self._aero.prescribed_wake else None,
        )

        u_n_aero_vec = self._aero._pack_input_vector(u_n_aero)
        x_n_aero_vec = self._aero._pack_state_vector(x_n_aero)

        x_np1_aero_vec, y_np1_aero_vec = self._aero.step_vec(
            x_vec=x_n_aero_vec, u_vec=u_n_aero_vec
        )

        x_np1_aero = self._aero._unpack_state_vector(x=x_np1_aero_vec)
        y_np1_aero = self._aero._unpack_output_vector(y=y_np1_aero_vec)
        assert isinstance(x_np1_aero, AeroStateUnflattened) and isinstance(
            y_np1_aero, AeroOutputUnflattened
        ), (
            "Unpacked aero state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        # total aero forces on the grid, from the aero step (returns totals)
        f_aero_np1 = y_np1_aero.f_steady
        if self.unsteady_force:
            assert y_np1_aero.f_unsteady is not None
            f_aero_np1 += y_np1_aero.f_unsteady

        # project total aero forces onto the beam under the current (perturbed) rotation
        rmat = hg[:, :3, :3]
        f_aero_beam_total = project_forcing_to_beam(
            f_total=f_aero_np1,
            rmat=rmat,
            dof_mapping=self._aero.case.dof_mapping,
            x0_aero=self._aero.case.x0_b,
        )

        # subtract the reference contribution so the aero forcing fed into the beam operator
        # is a pure perturbation (the beam sys.a / sys.b operate on perturbations)
        f_aero_ref_total = ref.aero.f_steady
        if self._aero.unsteady_force:
            f_aero_ref_total += ref.aero.f_unsteady
        f_aero_beam_ref = project_forcing_to_beam(
            f_total=f_aero_ref_total,
            rmat=ref.structure.hg[:, :3, :3],
            dof_mapping=self._aero.case.dof_mapping,
            x0_aero=self._aero.case.x0_b,
        )
        delta_f_aero_beam = f_aero_beam_total - f_aero_beam_ref

        f_ext_dead_: Array = (
            f_ext_dead if f_ext_dead is not None else jnp.zeros(self.n_beam_dof)
        )
        f_ext_follower_: Array = (
            f_ext_follower if f_ext_follower is not None else jnp.zeros(self.n_beam_dof)
        )

        # scatter free-dof forcing to full (n_nodes * 6) vectors expected by the beam B operator, and add aero
        # forcing (global frame, perturbation) into the dead-force channel
        f_ext_dead_full = (
            jnp.zeros(self.n_nodes * 6).at[self.free_dofs].set(f_ext_dead_)
            + delta_f_aero_beam.ravel()
        )
        f_ext_follower_full = (
            jnp.zeros(self.n_nodes * 6).at[self.free_dofs].set(f_ext_follower_)
        )

        # beam step in perturbation form (discrete-time Tustin)
        x_beam_n = jnp.concatenate([q, q_dot])

        x_beam_np1 = (
            self._beam.sys.a @ x_beam_n
            + self._beam.sys.b[:, self._beam.input_slices["f_ext_dead"].slices]
            @ f_ext_dead_full
            + self._beam.sys.b[:, self._beam.input_slices["f_ext_follower"].slices]
            @ f_ext_follower_full
        )

        q_np1 = x_beam_np1[: self.n_beam_dof]
        q_dot_np1 = x_beam_np1[self.n_beam_dof :]

        state_np1 = AeroelasticStateUnflattened(
            gamma_b=x_np1_aero.gamma_b,
            gamma_w=x_np1_aero.gamma_w,
            gamma_b_nm1=x_np1_aero.gamma_b_nm1,
            zeta_w=x_np1_aero.zeta_w,
            q=q_np1,
            q_dot=q_dot_np1,
        )
        output_n = AeroelasticOutputUnflattened(q=q, q_dot=q_dot)

        return state_np1, output_n

    def gamma_b_step(
        self,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        q_n: Array,
        q_dot_n: Array,
        zeta_w_n_vec: Optional[Array] = None,
        nu_b_n_vec: Optional[Array] = None,
    ) -> Array:
        r"""
        Bound circulation at timestep n+1 as a function of states at n and inputs at n+1.
        """
        x_np1, _ = self.step(
            nu_b_vec=nu_b_n_vec,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q=q_n,
            q_dot=q_dot_n,
        )
        return x_np1.gamma_b.ravel()

    def wake_prop_step(
        self,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        q_n: Array,
        zeta_w_n_vec: Optional[Array] = None,
        nu_w_n_vec: Optional[Array] = None,
    ) -> tuple[Optional[Array], Array]:
        r"""
        Wake propagation as a function of states at n and inputs at n+1.
        """
        x_new, _ = self.step(
            nu_w_vec=nu_w_n_vec,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q=q_n,
        )
        assert not ((x_new.zeta_w is not None) ^ self._aero.prescribed_wake), (
            "zeta_w should be None only if prescribed_wake is False."
        )
        return (
            x_new.zeta_w.ravel() if x_new.zeta_w is not None else None,
            x_new.gamma_w.ravel(),
        )

    def q_step(
        self,
        q_n: Array,
        q_dot_n: Array,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        zeta_w_n_vec: Optional[Array] = None,
    ) -> Array:
        r"""
        Beam displacement at timestep n+1 as a function of states at n and external forcing.
        """
        x_new, _ = self.step(
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q=q_n,
            q_dot=q_dot_n,
        )
        return x_new.q.ravel()

    def q_dot_step(
        self,
        q_n: Array,
        q_dot_n: Array,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        zeta_w_n_vec: Optional[Array],
    ) -> Array:
        r"""
        Beam velocity at timestep n+1 as a function of states at n and external forcing.
        """
        x_new, _ = self.step(
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q=q_n,
            q_dot=q_dot_n,
        )
        return x_new.q_dot.ravel()

    def compute_jacobians(
        self,
    ) -> tuple[
        dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]],
        dict[str, dict[str, Callable[..., Array]]],
    ]:
        r"""
        :return: Tuple. First entry is dictionaries with keys being the function name (e.g., gamma_b, gamma_w), with
        each entry containing the relevant stepping function, the arguments for the function, and the name of the
        arguments for which to obtain derivatives. Second entry is a dictionary of explicit Jacobians functions that
        take the same arguments as the first output.
        """
        ref = self.reference

        # bound circulation
        gamma_b_args: dict[str, Any] = dict(
            gamma_b_n_vec=ref.aero.gamma_b.ravel(),
            gamma_w_n_vec=ref.aero.gamma_w.ravel(),
            q_n=jnp.zeros((self.n_beam_dof,)),
            q_dot_n=jnp.zeros((self.n_beam_dof,)),
            zeta_w_n_vec=ref.aero.zeta_w.ravel(),
            nu_b_n_vec=jnp.zeros(ref.aero.zeta_b.size)
            if self._aero.bound_upwash
            else None,
        )
        gamma_b_diff = ["gamma_w_n_vec", "q_n", "q_dot_n"]
        if self._aero.prescribed_wake:
            gamma_b_diff.append("zeta_w_n_vec")
        if self._aero.bound_upwash:
            gamma_b_diff.append("nu_b_n_vec")

        # wake
        wake_args: dict[str, Any] = dict(
            gamma_b_n_vec=ref.aero.gamma_b.ravel(),
            gamma_w_n_vec=ref.aero.gamma_w.ravel(),
            q_n=jnp.zeros((self.n_beam_dof,)),
            zeta_w_n_vec=ref.aero.zeta_w.ravel(),
            nu_w_n_vec=jnp.zeros(ref.aero.zeta_w.size)
            if self._aero.wake_upwash
            else None,
        )
        gamma_w_diff = ["gamma_b_n_vec", "gamma_w_n_vec"]
        zeta_w_diff = ["q_n"]
        if self._aero.prescribed_wake:
            zeta_w_diff.append("zeta_w_n_vec")
        if self._aero.wake_upwash:
            zeta_w_diff.append("nu_w_n_vec")
        if self._aero.free_wake:
            zeta_w_diff.extend(["gamma_b_n_vec", "gamma_w_n_vec"])

        q_args: dict[str, Any] = dict(
            q_n=jnp.zeros((self.n_beam_dof,)),
            q_dot_n=jnp.zeros((self.n_beam_dof,)),
            gamma_b_n_vec=ref.aero.gamma_b.ravel(),
            gamma_w_n_vec=ref.aero.gamma_w.ravel(),
            f_ext_follower=jnp.zeros((self.n_beam_dof,)),
            f_ext_dead=jnp.zeros((self.n_beam_dof,)),
            zeta_w_n_vec=ref.aero.zeta_w.ravel(),
        )
        q_diff = [
            "q_n",
            "q_dot_n",
            "gamma_b_n_vec",
            "gamma_w_n_vec",
            "f_ext_follower",
            "f_ext_dead",
        ]

        if self._aero.prescribed_wake:
            q_diff.append("zeta_w_n_vec")

        linear_args: dict[
            str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]
        ] = {
            "gamma_b": (self.gamma_b_step, gamma_b_args, gamma_b_diff),
            "gamma_w": (
                lambda *args, **kwargs: self.wake_prop_step(**kwargs)[1],
                wake_args,
                gamma_w_diff,
            ),
            "gamma_b_nm1": (
                lambda *args, **kwargs: None,
                {
                    "gamma_b_n_vec": ref.aero.gamma_b.ravel(),
                },
                ["gamma_b_n_vec"],
            ),
            "q": (self.q_step, q_args, q_diff),
            "q_dot": (self.q_dot_step, q_args, q_diff),
        }

        # add zeta_w for linearisation
        if self._aero.prescribed_wake:
            linear_args["zeta_w"] = (
                lambda *args, **kwargs: self.wake_prop_step(**kwargs)[0],
                wake_args,
                zeta_w_diff,
            )

        # define Jacobians we know nicely as jac_options
        jac_options = {
            "q": {
                "q_n": lambda *args, **kwargs: self._beam.sys.a[
                    : self.n_beam_dof, : self.n_beam_dof
                ],
                "q_dot_n": lambda *args, **kwargs: self._beam.sys.a[
                    : self.n_beam_dof, self.n_beam_dof :
                ],
            },
            "q_dot": {
                "q_n": lambda *args, **kwargs: self._beam.sys.a[
                    self.n_beam_dof :, : self.n_beam_dof
                ],
                "q_dot_n": lambda *args, **kwargs: self._beam.sys.a[
                    self.n_beam_dof :, self.n_beam_dof :
                ],
            },
        }
        if self._aero.unsteady_force:
            jac_options["gamma_b_nm1"] = {
                "gamma_b_n_vec": lambda *args, **kwargs: jnp.eye(ref.aero.gamma_b.size)
            }

        return linear_args, jac_options

    def create_jacobians(
        self,
        n_profile_loops: Optional[int] = None,
        jac_options: Optional[
            dict[str, dict[str, Optional[Callable[..., Any]]]]
        ] = None,
    ) -> tuple[
        dict[str, dict[str, Array]],
        Optional[dict[str, dict[str, float]]],
        Optional[dict[str, dict[str, float]]],
    ]:
        res_args, jac_options_exp = self.compute_jacobians()
        jac_options_total: dict[str, dict[str, Optional[Callable[..., Any]]]] = (
            jac_options if jac_options is not None else {}
        ) | jac_options_exp

        jacobians: dict[str, dict[str, Array]] = {}
        compile_time: dict[str, dict[str, float]] = {}
        run_time: dict[str, dict[str, float]] = {}

        for res_name, (res_func, args, diff_arg_names) in res_args.items():
            res_jac_options: dict[str, Optional[Callable[..., Any]]] = {
                arg: None for arg in diff_arg_names
            }
            if res_name in jac_options_total:
                for arg, entry in jac_options_total[res_name].items():
                    if arg in res_jac_options:
                        res_jac_options[arg] = entry

            jacs, res_compile_time, res_run_time = jacrev_custom(
                func=res_func,
                jac_options=res_jac_options,
                n_profile_loops=n_profile_loops,
                func_name=res_name,
            )(**args)

            jacobians[res_name] = jacs
            if n_profile_loops is not None:
                assert res_compile_time is not None and res_run_time is not None
                compile_time[res_name] = res_compile_time
                run_time[res_name] = res_run_time

        return (
            jacobians,
            compile_time if n_profile_loops is not None else None,
            run_time if n_profile_loops is not None else None,
        )

    def linearise(self) -> LinearSystem:
        jacobians, _, _ = self.create_jacobians(n_profile_loops=None, jac_options=None)

        # (row-label in `jacobians`, column-argname used inside each jacobian dict, size)
        state_specs: list[tuple[str, str, int]] = [
            ("gamma_b", "gamma_b_n_vec", self.reference.aero.gamma_b.size),
            ("gamma_w", "gamma_w_n_vec", self.reference.aero.gamma_w.size),
        ]
        if self._aero.unsteady_force:
            state_specs.append(
                ("gamma_b_nm1", "gamma_b_nm1_vec", self.reference.aero.gamma_b.size)
            )
        if self._aero.prescribed_wake:
            state_specs.append(
                ("zeta_w", "zeta_w_n_vec", self.reference.aero.zeta_w.size)
            )
        state_specs.extend(
            [
                ("q", "q_n", self.n_beam_dof),
                ("q_dot", "q_dot_n", self.n_beam_dof),
            ]
        )
        state_names = [s[0] for s in state_specs]
        state_arg_names = [s[1] for s in state_specs]
        state_sizes = [s[2] for s in state_specs]

        input_specs: list[tuple[str, int]] = []
        if self._aero.bound_upwash:
            input_specs.append(("nu_b_n_vec", self.reference.aero.zeta_b.size))
        if self._aero.wake_upwash:
            input_specs.append(("nu_w_n_vec", self.reference.aero.zeta_w.size))
        input_specs.extend(
            [
                ("f_ext_follower", self.n_beam_dof),
                ("f_ext_dead", self.n_beam_dof),
            ]
        )
        input_arg_names = [s[0] for s in input_specs]
        input_sizes = [s[1] for s in input_specs]

        output_sizes = [self.n_beam_dof, self.n_beam_dof]

        a = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in state_names),
            keys=state_arg_names,
            widths=state_sizes,
            heights=state_sizes,
        )

        b = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in state_names),
            keys=input_arg_names,
            widths=input_sizes,
            heights=state_sizes,
        )

        c_entries: tuple[dict[str, Array], ...] = (
            {"q_n": jnp.eye(self.n_beam_dof)},
            {"q_dot_n": jnp.eye(self.n_beam_dof)},
        )
        c = construct_named_block_jacobian(
            entries=c_entries,
            keys=state_arg_names,
            widths=state_sizes,
            heights=output_sizes,
        )
        d = jnp.zeros((sum(output_sizes), sum(input_sizes)))

        return LinearSystem(a=a, b=b, c=c, d=d, dt=self.dt, removed_u_np1=False)

    def linearise_profile(
        self,
        n_profile_loops: int = 10,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        r"""
        Profile forming the Jacobians required for the linearised model.
        :param n_profile_loops: Number of times to loop Jacobian creation for averaging.
        :return: Dictionaries of compile and run times for each sub function.
        """

        print_table_title(inner_width=95, title="Aeroelastic Adjoint Profile")

        _, compile_time, run_time = self.create_jacobians(
            n_profile_loops=n_profile_loops, jac_options=None
        )

        assert compile_time is not None and run_time is not None, (
            "No output timings passed"
        )

        print_table_line(inner_width=95)

        return compile_time, run_time

    # noinspection PyMethodOverriding
    def run(
        self,
        u: AeroelasticInputUnflattened,
        x0: Optional[AeroelasticStateUnflattened] = None,
    ) -> AeroelasticLinearResult:
        raise NotImplementedError
