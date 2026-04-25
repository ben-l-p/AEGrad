import jax
from jax import Array, numpy as jnp

from aegrad.utils.print_utils import set_verbosity
from aegrad.structure import StructureFullStates
from aegrad.utils.data_structures import ConvergenceSettings
from algebra.so3 import log_so3
from models.geradin_beam import geradin_beam
from structure import BeamStructure, StaticStructure
from utils.print_utils import VerbosityLevel

if __name__ == "__main__":
    """
    Obtain the deformation of the Geradin beam undergoing a tip vertical load, and the gradient of the tip node 
    z-coordinate with respect to the tip load magnitude and the bending stiffness magnitude. This case verifies 
    gradients with finite differences.
    """
    jax.config.update("jax_enable_x64", True)
    set_verbosity(VerbosityLevel.SILENT)

    n_nodes = 20
    struct = geradin_beam(n_nodes, "x_target")

    # convergence very strict, forces 100 structural iterations
    struct.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=50,
        rel_force_tol=0.0,
        rel_disp_tol=0.0,
        abs_force_tol=0.0,
        abs_disp_tol=0.0,
    )
    load = 600000.0
    f_ext = jnp.zeros((n_nodes, 6)).at[-1, 2].set(-load)

    def _solve(struct_: BeamStructure, f_ext_: Array) -> StaticStructure:
        return struct_.static_solve(
            f_ext_follower=None,
            f_ext_dead=f_ext_,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
            load_steps=3,
        )

    def _objective(states: StructureFullStates, *_) -> Array:
        return states.hg[-1, 2, 3]  # tip vertical displacement

    # solve adjoint system
    base_result = _solve(struct, f_ext)
    grads_adj = struct.static_adjoint(structure=base_result, objective=_objective)
    obj_base = _objective(base_result.get_full_states())

    print("Primal solution")
    print(f"Tip z displacement: {base_result.hg[-1, 2, 3]}")
    rot = jnp.rad2deg(-log_so3(base_result.hg[-1, :3, :3])[1])
    print(f"Tip rotation angle: {rot}")

    # solve for perturbed force
    f_eps = 1.0

    f_pert = f_ext.at[-1, 2].add(f_eps)
    f_pert_obj = _objective(_solve(struct, f_pert).get_full_states())

    if grads_adj.f_ext_dead is None:
        raise ValueError("f_ext_dead is None")
    f_fd_grad = (f_pert_obj - obj_base) / f_eps
    print("\nTip forcing gradient")
    print(f"Adjoint: {grads_adj.f_ext_dead[-1, 2]}")
    print(f"Finite difference: {f_fd_grad}")

    # solve for perturbed bending stiffness
    k_eps = 10.0
    struct.k_cs = struct.k_cs.at[0, 4, 4].add(k_eps)
    k_pert_obj = _objective(_solve(struct, f_ext).get_full_states())
    k_fd_grad = (k_pert_obj - obj_base) / k_eps
    if grads_adj.k_cs is None:
        raise ValueError("k_cs is None")
    print("\nBeam bending stiffness gradient")
    print(f"Adjoint: {grads_adj.k_cs[0, 4, 4]}")
    print(f"Finite difference: {k_fd_grad}")
