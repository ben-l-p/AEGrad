import jax
from jax import numpy as jnp
from jax import Array

from structure import BeamStructure
from structure import StructureFullStates, StructuralDesignVariables
from models.geradin_beam import geradin_beam

jax.config.update("jax_enable_x64", True)


class TestGeradinBeamAdjointGradients:
    n_nodes = 20
    struct = geradin_beam(n_nodes, "x_target")
    load = 600000.0
    f_ext = jnp.zeros((n_nodes, 6)).at[-1, 2].set(-load)

    @classmethod
    def test_gradient(cls):
        struct = geradin_beam()

        # primal solve
        result = struct.static_solve(
            f_ext_follower=None,
            f_ext_dead=cls.f_ext,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
            load_steps=3,
        )

        def obj(
                states: StructureFullStates,
                *_,
        ) -> Array:
            return states.hg[-1, 2, 3]  # vertical displacement of the last node

        # grads solve
        grads_adj = struct.static_adjoint(structure=result, objective=obj)

        # AD solve
        def resolve_obj(dv_: StructuralDesignVariables) -> Array:
            n_elem = cls.n_nodes - 1

            y_vect = jnp.array([0.0, 1.0, 0.0])

            conn = jnp.zeros((n_elem, 2), dtype=int)
            conn = conn.at[:, 0].set(jnp.arange(n_elem))
            conn = conn.at[:, 1].set(jnp.arange(1, n_elem + 1))

            struct_ = BeamStructure(cls.n_nodes, conn, y_vect[None, :])

            struct_.set_design_variables(dv_.x0, dv_.k_cs, dv_.m_cs)

            result_ = struct_.static_solve(
                f_ext_follower=None,
                f_ext_dead=cls.f_ext,
                f_ext_aero=None,
                prescribed_dofs=jnp.arange(6),
                load_steps=3,
            )

            ss = StructureFullStates(
                hg=result_.hg,
                eps=result_.eps,
                f_int=result_.f_int,
                v=None,
                v_dot=None,
            )

            return obj(ss, dv_, None)

        dv = StructuralDesignVariables(
            x0=struct.x0,
            k_cs=struct.k_cs,
            m_cs=struct.m_cs,
            m_lumped=struct.m_lumped,
            f_ext_dead=cls.f_ext,
            f_ext_follower=None,
        )

        # ideally this would be jacrev, but reverse-mode AD is currently unsupported
        grads_ad = jax.jacfwd(resolve_obj)(dv)
        assert grads_adj.f_ext_dead is not None, "Gradient w.r.t. external forces is None"
        assert jnp.allclose(grads_adj.f_ext_dead, grads_ad.f_ext_dead, atol=2e-6), (
            "Gradient w.r.t. external forces does not match"
        )
        assert jnp.allclose(grads_adj.k_cs, grads_ad.k_cs), (
            "Gradient w.r.t. cross-sectional stiffness does not match"
        )
