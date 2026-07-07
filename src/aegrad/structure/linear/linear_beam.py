from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Optional

from jax import Array, vmap
from jax import numpy as jnp

from aegrad.algebra.array_utils import check_arr_shape
from aegrad.algebra.linear_operators import LinearSystem, LinearOperator
from aegrad.algebra.se3 import exp_se3
from aegrad.structure.utils import get_solve_dofs, transform_nodal_vect
from aegrad.utils.constants import BASE_LOBATTO_ORDER
from aegrad.structure.data_structures import StaticStructure

if TYPE_CHECKING:
    from aegrad.structure.beam import BaseBeamStructure


class BeamLinearResult:
    def __init__(
        self,
        reference: StaticStructure,
        f_ext_follower: Optional[Array],
        f_ext_dead: Optional[Array],
        delta_q: Array,
        delta_q_dot: Array,
        hg: Array,
        t: Array,
    ) -> None:
        # system results, if simulated
        self.f_ext_follower: Optional[Array] = f_ext_follower
        self.f_ext_dead: Optional[Array] = f_ext_dead
        self.delta_q: Array = delta_q
        self.delta_q_dot: Array = delta_q_dot
        self.hg: Array = hg
        self.n_tstep: int = len(t)
        self.t: Array = t
        self.reference: StaticStructure = reference


class LinearBeam:
    r"""
    Class to represent a linearised beam system about a reference state.
    """

    def __init__(
        self,
        beam: BaseBeamStructure,
        reference: StaticStructure,
        dt: float | Array,
        local_forcing: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
    ):
        self.n_dof: int = beam.n_dof - len(reference.prescribed_dofs)
        self.n_nodes: int = beam.n_nodes
        self.free_dofs: Array = jnp.array(
            get_solve_dofs(n_dof=beam.n_dof, prescribed_dofs=reference.prescribed_dofs)
        )

        self.reference: StaticStructure = reference
        self.local_forcing: bool = local_forcing

        self.m_global, self.k_global = beam.make_global_m_k(
            case=reference, int_order=int_order, local_forcing=local_forcing
        )

        self.dt: float = float(dt)

        self.sys: LinearSystem = self.linearise_continuous()

    def linearise_continuous(self) -> LinearSystem:
        r"""
        Form a system of linear equations about a reference state. The system is of the form:
        :math:`\dot{\mathbf{x}} = \mathbf{A~x + B~u}, \mathbf{y} = \mathbf{C~x + D~u}`. Input vector :math:`\mathbf{u}`
        is the vector of applied forces, output vector :math:`\mathbf{y}` is the vector of displacements, and state
        vector :math:`\mathbf{x}` is the vector of displacements and velocities.
        :return: Linearised system.
        """

        m_inv = jnp.linalg.inv(self.m_global)

        a = LinearOperator(
            lambda x: jnp.concatenate(
                (x[self.n_dof :], -m_inv @ self.k_global @ x[: self.n_dof])
            ),
            shape=(2 * self.n_dof, 2 * self.n_dof),
        )

        b = LinearOperator(
            lambda u: jnp.concatenate((jnp.zeros((self.n_dof,)), m_inv @ u)),
            shape=(2 * self.n_dof, self.n_dof),
        )

        c = LinearOperator(
            lambda x: x[: self.n_dof], shape=(self.n_dof, 2 * self.n_dof)
        )

        d = LinearOperator(
            lambda u: jnp.zeros((self.n_dof,)), shape=(self.n_dof, self.n_dof)
        )

        return LinearSystem(
            a=a, b=b, c=c, d=d, dt=self.dt, continuous_time=True, removed_u_np1=False
        )

    def run(
        self,
        n_tstep: int,
        f_ext_follower_t: Optional[Array],
        f_ext_dead_t: Optional[Array],
        q0: Optional[Array] = None,
        q0_dot: Optional[Array] = None,
        use_matrix=False,
    ) -> BeamLinearResult:
        r"""
        Run the linear system.
        :param n_tstep. Number of time steps to simulate.
        :param f_ext_follower_t: External follower forces applied to the system at each time step, [n_tstep, n_nodes, 6].
        :param f_ext_dead_t: External dead forces applied to the system at each time step, [n_tstep, n_nodes, 6].
        :param q0: Initial displacements, [n_nodes, 6]. Set as None to use zero initial displacements.
        :param q0_dot: Initial velocities, [n_nodes, 6]. Set as None to use zero initial velocities.
        :param use_matrix: If true, use explicit matrix representation for linear system, otherwise use operator form.
        :return: Linear system results.
        """

        if q0 is None:
            q0 = jnp.zeros((self.n_dof,))
        else:
            q0 = q0.ravel()[self.free_dofs]
            check_arr_shape(q0, (self.n_dof,), name="q0")

        if q0_dot is None:
            q0_dot = jnp.zeros((self.n_dof,))
        else:
            q0_dot = q0_dot.ravel()[self.free_dofs]
            check_arr_shape(q0_dot, (self.n_dof,), name="q0_dot")

        def create_delta_forcing(
            reference_f: Optional[Array],
            input_f_t: Optional[Array],
            name: str,
            transformation: Optional[Array],
        ) -> Array:
            r"""
            Function to compute the forcing perturbations to pass to the linear system. This accounts for the reference
            forcing and any transformations to local or global coordinates.
            :param reference_f: Reference forcing vector, [n_nodes, 6] or None.
            :param input_f_t: Input forcing vector, [n_tstep, n_nodes, 6] or None.
            :param name: Name of the forcing vector for error messages.
            :param transformation: Transformation matrix for coordinate transformation or None.
            :return: Delta forcing vector, [n_tstep, n_nodes, 6].
            """
            if input_f_t is None:
                if reference_f is None:
                    return jnp.zeros((n_tstep, self.n_nodes, 6))
                else:
                    return -jnp.broadcast_to(
                        self.reference.f_ext_dead[None, :, :],
                        (n_tstep, self.n_nodes, 6),
                    )
            else:
                check_arr_shape(input_f_t, (n_tstep, self.n_nodes, 6), name=name)
                if transformation is not None:
                    f_t_tot = transform_nodal_vect(
                        vect=input_f_t,
                        rmat=jnp.swapaxes(self.reference.hg[:, :3, :3], -1, -2),
                    )
                else:
                    f_t_tot = input_f_t

                if reference_f is None:
                    return f_t_tot
                else:
                    return f_t_tot - reference_f[None, :, :]

        delta_f_ext_follower_t = create_delta_forcing(
            reference_f=self.reference.f_ext_follower,
            input_f_t=f_ext_follower_t,
            name="f_ext_follower_t",
            transformation=None if self.local_forcing else self.reference.hg[:, :3, :3],
        )

        delta_f_ext_dead_t = create_delta_forcing(
            reference_f=self.reference.f_ext_dead,
            input_f_t=f_ext_dead_t,
            name="f_ext_dead_t",
            transformation=jnp.swapaxes(self.reference.hg[:, :3, :3], -1, -2)
            if self.local_forcing
            else None,
        )

        delta_f_ext_t_free = (delta_f_ext_follower_t + delta_f_ext_dead_t).reshape(
            n_tstep, -1
        )[:, self.free_dofs]  # flatten to (n_tstep, n_dof)

        # run linear system
        x_t, y_t = self.sys.run(
            u=delta_f_ext_t_free,
            x0=jnp.concatenate((q0, q0_dot)),
            use_matrix=use_matrix,
        )

        # extract perturbations in displacements and velocities
        delta_q_t = x_t[:, : self.n_dof]
        delta_q_dot_t = x_t[:, self.n_dof :]

        # add in zeros for dofs not solved for to allow for reconstructing the full configuration
        delta_q_t_full = (
            jnp.zeros((n_tstep, self.n_nodes * 6))
            .at[:, self.free_dofs]
            .set(delta_q_t)
            .reshape(n_tstep, self.n_nodes, 6)
        )

        delta_hg_t = vmap(vmap(exp_se3, 0, 0), 1, 1)(
            delta_q_t_full
        )  # [n_tstep, n_nodes, 4, 4]

        hg_t = jnp.einsum("ijk,hikl->hijl", self.reference.hg, delta_hg_t)

        return BeamLinearResult(
            reference=self.reference,
            f_ext_follower=f_ext_follower_t,
            f_ext_dead=f_ext_dead_t,
            delta_q=delta_q_t,
            delta_q_dot=delta_q_dot_t,
            hg=hg_t,
            t=jnp.arange(n_tstep) * self.dt,
        )
