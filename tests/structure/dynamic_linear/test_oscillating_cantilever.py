from jax import numpy as jnp, vmap

from aegrad.algebra.se3 import log_se3
from aegrad.structure import BeamStructure
from aegrad.structure.linear.data_structures import BeamInputUnflattened


class TestOscillatingCantileverDeadLocal:
    n_nodes = 10
    length = 1.0
    m_bar = 1.0
    j_bar = 0.1
    ea, ga, gj, eay, eaz = 1e4, 1e4, 1e4, 10.0, 10.0

    k_cs = jnp.diag(jnp.array((ea, ga, ga, gj, eay, eaz)))
    m_cs = jnp.diag(jnp.array((m_bar, m_bar, m_bar, j_bar, j_bar, j_bar)))

    coords = jnp.zeros((n_nodes, 3))
    coords = coords.at[:, 0].set(jnp.linspace(0.0, length, n_nodes))

    conn = jnp.zeros((n_nodes - 1, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_nodes - 1))
    conn = conn.at[:, 1].set(jnp.arange(1, n_nodes))

    y_vector = jnp.array((0.0, 1.0, 0.0))

    beam = BeamStructure(
        num_nodes=n_nodes, connectivity=conn, y_vector=y_vector, spectral_radius=1.0
    )

    beam.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)

    dt = 0.01
    physical_time = 2.0
    n_tstep = int(physical_time / dt)

    is_dead: bool = True
    local_force_linearisation: bool = True

    @classmethod
    def test_oscillating_undeform_cantilever(cls):
        r"""
        Test the oscillating cantilever beam with a tip load. The cantilever begins in an undeformed reference and is
        subject to a vertical tip load, which causes it to oscillate. The test compares the nonlinear and linearised
        solutions of the beam's dynamic response.
        """

        ref = cls.beam.reference_configuration(
            use_f_grav=False,
            use_f_aero=False,
            use_f_ext_dead=cls.is_dead,
            use_f_ext_follower=not cls.is_dead,
            prescribed_dofs=tuple(range(6)),
        )

        linear_beam = cls.beam.linearise(
            reference=ref, dt=cls.dt, local_forcing=cls.local_force_linearisation
        )

        # nonlinear case
        f_beam = jnp.zeros((cls.n_tstep, cls.n_nodes, 6))
        f_beam = f_beam.at[:, -1, 2].set(1.0)  # tip load

        nl_sol = cls.beam.dynamic_solve(
            init_state=None,
            prescribed_dofs=tuple(range(6)),
            f_ext_dead=f_beam if cls.is_dead else None,
            f_ext_follower=None if cls.is_dead else f_beam,
            n_tstep=cls.n_tstep,
            dt=cls.dt,
        )

        u_beam = BeamInputUnflattened(
            n_tstep=cls.n_tstep,
            f_ext_follower=None if cls.is_dead else f_beam,
            f_ext_dead=f_beam if cls.is_dead else None,
        )

        lin_sol = linear_beam.run(
            u=u_beam,
        )

        nl_tip_z = nl_sol.x[:, -1, 2]
        lin_tip_z = lin_sol.x[:, -1, 2]

        # chosen tolerances that are considerably smaller than the maximum displacements and rotations
        assert jnp.allclose(nl_tip_z, lin_tip_z, atol=6e-4), (
            "Nonlinear and linear tip displacements do not match"
        )

        # can extract angles as there is only one nonzero rotational component
        nl_tip_rot = nl_sol.varphi[:, -1, 4]
        lin_tip_rot = lin_sol.delta_q[:, -2]

        assert jnp.allclose(nl_tip_rot, lin_tip_rot, atol=8e-4), (
            "Nonlinear and linear tip rotations do not match"
        )

    @classmethod
    def test_oscillating_deformed_cantilever(cls):
        r"""
        Test the oscillating cantilever beam with a tip load. The cantilever begins in a deformed reference and is
        subject to a vertical tip load, which causes it to oscillate. The test compares the nonlinear and linearised
        solutions of the beam's dynamic response.
        """

        f_ext_base = jnp.zeros((cls.n_nodes, 6)).at[-1, 2].set(3.0)  # static tip load
        f_ext_perturb = (
            jnp.zeros((cls.n_tstep, cls.n_nodes, 6)).at[:, -1, 2].set(3.0 + 0.01)
        )  # tip load perturbation

        ref = cls.beam.static_solve(
            prescribed_dofs=tuple(range(6)),
            f_ext_dead=f_ext_base if cls.is_dead else None,
            f_ext_follower=None if cls.is_dead else f_ext_base,
        )

        linear_beam = cls.beam.linearise(reference=ref, dt=cls.dt)

        # nonlinear case
        nl_sol = cls.beam.dynamic_solve(
            init_state=ref,
            prescribed_dofs=tuple(range(6)),
            f_ext_dead=f_ext_perturb if cls.is_dead else None,
            f_ext_follower=None if cls.is_dead else f_ext_perturb,
            n_tstep=cls.n_tstep,
            dt=cls.dt,
        )

        u_beam = BeamInputUnflattened(
            n_tstep=cls.n_tstep,
            f_ext_dead=f_ext_perturb if cls.is_dead else None,
            f_ext_follower=None if cls.is_dead else f_ext_perturb,
        )

        lin_sol = linear_beam.run(
            u=u_beam,
        )

        nl_tip_z = nl_sol.x[:, -1, 2]
        lin_tip_z = lin_sol.x[:, -1, 2]

        # chosen tolerances that are considerably smaller than the maximum displacements and rotations
        assert jnp.allclose(nl_tip_z, lin_tip_z, atol=7e-5), (
            "Nonlinear and linear tip displacements do not match"
        )

        nl_tip_rot = nl_sol.varphi[:, -1, 4]
        lin_tip_rot = vmap(lambda hg: log_se3(hg)[4])(lin_sol.hg[:, -1, ...])

        assert jnp.allclose(nl_tip_rot, lin_tip_rot, atol=7e-5), (
            "Nonlinear and linear tip rotations do not match"
        )


class TestOscillatingCantileverFollowerLocal(TestOscillatingCantileverDeadLocal):
    is_dead: bool = False


class TestOscillatingCantileverDeadGlobal(TestOscillatingCantileverDeadLocal):
    local_force_linearisation: bool = False


class TestOscillatingCantileverFollowerGlobal(TestOscillatingCantileverDeadLocal):
    is_dead: bool = False
    local_force_linearisation: bool = False
