from jax import numpy as jnp

from flapjax.structure import BeamStructure


class TestFreeFreeModes:
    n_nodes = 40
    n_elem = n_nodes - 1
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

    coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(0.0, 2.0, n_nodes))
    y_vect = jnp.zeros((n_elem, 3)).at[:, 1].set(1.0)
    struct = BeamStructure(num_nodes=n_nodes, connectivity=conn, y_vector=y_vect)

    struct.set_design_variables(coords=coords, k_cs=1e6 * jnp.eye(6), m_cs=jnp.eye(6))

    ref_case = struct.reference_configuration(
        use_f_ext_follower=False,
        use_f_ext_dead=False,
        use_f_aero=False,
        use_f_grav=False,
        prescribed_dofs=(),  # free structure
    )

    @classmethod
    def test_rigid_body_mode_frequencies(cls):
        r"""
        The first six modes of a free structure should be rigid body modes, with frequencies negligible compared to
        the first elastic mode.
        """
        frequencies, *_ = cls.struct.modal(case=cls.ref_case, n_modes=10)

        assert jnp.all(frequencies[:6] < 2e-4 * frequencies[6]), (
            f"Expected six rigid body modes with negligible frequency, got {frequencies[:6]}"
        )

    @classmethod
    def test_rigid_body_modes_have_no_strain_energy(cls):
        r"""
        Rigid body mode shapes should lie in the null space of the stiffness matrix, and therefore have negligible
        strain energy.
        """
        _, _, modes, _, k_nodal = cls.struct.base_modal(
            case=cls.ref_case, remove_complex_conjugate=False, n_modes=10
        )

        rb_modes = modes[:6, :]
        strain_energy = jnp.einsum("mi,ij,mj->m", rb_modes, k_nodal, rb_modes)

        assert jnp.all(strain_energy < 2e-6), (
            f"Rigid body modes have non-zero strain energy: {strain_energy}"
        )

    @classmethod
    def test_rigid_body_modes_span_translations_and_rotations(cls):
        r"""
        Ensure the six rigid body mode shapes are pure translations and rotations about the origin.
        """
        _, _, rb_modes, m_nodal, _ = cls.struct.base_modal(
            case=cls.ref_case, remove_complex_conjugate=False, n_modes=6
        )

        # canonical rigid body modes:
        # three translational modes
        # three rotational modes about the origin (non-axial rotations also have a translational component).
        rb_canonical = jnp.zeros((6, cls.n_nodes * 6))
        for i in range(3):
            # translational modes
            rb_canonical = rb_canonical.at[i, i::6].set(1.0)

        r = cls.coords

        # rotation about beam (no translational component as lies on the axis)
        rb_canonical = rb_canonical.at[3, 3::6].set(1.0)

        # out-of-axis rotations
        rb_canonical = rb_canonical.at[4, 0::6].set(r[:, 2])
        rb_canonical = rb_canonical.at[4, 2::6].set(-r[:, 0])
        rb_canonical = rb_canonical.at[4, 4::6].set(1.0)

        rb_canonical = rb_canonical.at[5, 0::6].set(-r[:, 1])
        rb_canonical = rb_canonical.at[5, 1::6].set(r[:, 0])
        rb_canonical = rb_canonical.at[5, 5::6].set(1.0)

        coeffs = jnp.einsum("mi,ij,cj->cm", rb_modes, m_nodal, rb_canonical)
        reconstructed = coeffs @ rb_modes

        # compare against canonical modes
        residual = rb_canonical - reconstructed

        assert jnp.allclose(residual, 0.0, atol=3e-7), (
            f"Computed rigid body modes do not match canonical modes, residual max: {jnp.max(jnp.abs(residual))}"
        )
