from copy import deepcopy

from jax import Array, numpy as jnp
from aegrad.structure import StructureFullStates
from aegrad.utils.data_structures import ConvergenceSettings
from models.geradin_beam import geradin_beam


class TestGeradinBeamGradients:
    n_nodes = 20
    struct = geradin_beam(n_nodes, "x_target")
    struct.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=50,
        rel_force_tol=0.0,
        rel_disp_tol=0.0,
        abs_force_tol=0.0,
        abs_disp_tol=0.0,
    )
    load = 600000.0
    f_ext = jnp.zeros((n_nodes, 6)).at[-1, 2].set(-load)

    @classmethod
    def _solve(cls, struct, f_ext):
        return struct.static_solve(
            f_ext_follower=None,
            f_ext_dead=f_ext,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
            load_steps=3,
        )

    @staticmethod
    def _objective(states: StructureFullStates, *_) -> Array:
        return states.hg[-1, 2, 3]  # tip vertical displacement

    @classmethod
    def test_stiffness_gradient(cls):
        r"""
        Check the adjoint gradient of tip displacement w.r.t. bending stiffness (eay)
        against a finite difference estimate.
        """
        result = cls._solve(cls.struct, cls.f_ext)
        grads_adj = cls.struct.static_adjoint(
            structure=result, objective=cls._objective
        )

        eps = 10.0  # large epsilon as stiffness values are large
        obj_base = cls._objective(result.get_full_states())

        k_cs_pert = cls.struct.k_cs.at[0, 4, 4].add(eps)
        struct_pert = deepcopy(cls.struct)
        struct_pert.set_design_variables(struct_pert.x0, k_cs_pert, None, None)
        obj_pert = cls._objective(cls._solve(struct_pert, cls.f_ext).get_full_states())

        fd_grad = (obj_pert - obj_base) / eps
        if grads_adj.k_cs is None:
            raise ValueError("k_cs is None")
        adj_grad = grads_adj.k_cs[0, 4, 4]

        err = abs(fd_grad - adj_grad) / abs(adj_grad)
        assert err < 1e-5, (
            f"Stiffness gradient relative error {err:.2e} exceeds tolerance"
        )

    @classmethod
    def test_force_gradient(cls):
        r"""
        Check the adjoint gradient of tip displacement w.r.t. the applied tip load
        against a finite difference estimate.
        """
        result = cls._solve(cls.struct, cls.f_ext)
        grads_adj = cls.struct.static_adjoint(
            structure=result, objective=cls._objective
        )

        eps = 1.0
        obj_base = cls._objective(result.get_full_states())

        f_pert = cls.f_ext.at[-1, 2].add(eps)
        obj_pert = cls._objective(
            cls._solve(deepcopy(cls.struct), f_pert).get_full_states()
        )

        fd_grad = (obj_pert - obj_base) / eps
        if grads_adj.f_ext_dead is None:
            raise ValueError("Missing f_ext_dead gradient")
        adj_grad = grads_adj.f_ext_dead[-1, 2]

        err = abs(fd_grad - adj_grad) / abs(adj_grad)
        assert err < 1e-5, f"Force gradient relative error {err:.2e} exceeds tolerance"
